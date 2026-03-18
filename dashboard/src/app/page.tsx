"use client";

import { useState, useMemo } from "react";
import { TrendingUp, TrendingDown, Wallet, BarChart3, PieChart, Activity } from "lucide-react";
import type { Position } from "@/types";
import {
  formatCurrency,
  formatPct,
  formatDateTime,
  getPnlColor,
  getConfidenceColor,
} from "@/lib/utils";
import {
  mockPortfolioSummary,
  mockPositions,
  mockSectorAllocation,
  mockStrategyAllocation,
  mockHorizonAllocation,
  mockDirectionAllocation,
  mockRiskSummary,
  mockLastUpdate,
} from "@/data/mock";
import { MetricCard } from "@/components/shared/MetricCard";
import { DataTable, type Column } from "@/components/shared/DataTable";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { AllocationPanel } from "@/components/portfolio/AllocationPanel";
import { PositionDetail } from "@/components/portfolio/PositionDetail";

const summary = mockPortfolioSummary;

export default function PortfolioDashboard() {
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const openDetail = (pos: Position) => {
    setSelectedPosition(pos);
    setDrawerOpen(true);
  };

  const sortedByPnl = useMemo(
    () => [...mockPositions].sort((a, b) => b.unrealizedPnl - a.unrealizedPnl),
    []
  );
  const topWinners = sortedByPnl.slice(0, 3);
  const topLosers = sortedByPnl.slice(-3).reverse();

  const topPositionPct = useMemo(() => {
    const maxMv = Math.max(...mockPositions.map((p) => p.marketValue));
    return maxMv / summary.totalValue;
  }, []);

  const columns: Column<Position>[] = [
    {
      key: "symbol",
      header: "Symbol",
      className: "w-24",
      render: (p) => <span className="text-sm font-bold text-slate-800">{p.symbol}</span>,
    },
    {
      key: "direction",
      header: "Direction",
      className: "w-20",
      render: (p) => (
        <Badge variant={p.direction === "long" ? "success" : "danger"} size="sm">
          {p.direction.toUpperCase()}
        </Badge>
      ),
    },
    {
      key: "quantity",
      header: "Qty",
      className: "w-16",
      render: (p) => <span className="text-sm text-slate-500 tabular-nums">{p.quantity}</span>,
    },
    {
      key: "entry",
      header: "Entry",
      className: "w-24",
      render: (p) => <span className="text-sm text-slate-400 tabular-nums">${p.avgEntryPrice.toFixed(2)}</span>,
    },
    {
      key: "current",
      header: "Current",
      className: "w-24",
      render: (p) => <span className="text-sm text-slate-600 tabular-nums">${p.currentPrice.toFixed(2)}</span>,
    },
    {
      key: "mktValue",
      header: "Market Value",
      className: "w-28",
      render: (p) => <span className="text-sm text-slate-600 tabular-nums">{formatCurrency(p.marketValue)}</span>,
    },
    {
      key: "unrealPnl",
      header: "PnL",
      className: "w-28",
      render: (p) => (
        <span className={`text-sm font-semibold tabular-nums ${getPnlColor(p.unrealizedPnl)}`}>
          {formatCurrency(p.unrealizedPnl)}
        </span>
      ),
    },
    {
      key: "pnlPct",
      header: "PnL %",
      className: "w-20",
      render: (p) => (
        <span className={`text-sm tabular-nums ${getPnlColor(p.unrealizedPnlPct)}`}>
          {formatPct(p.unrealizedPnlPct)}
        </span>
      ),
    },
    {
      key: "strategy",
      header: "Strategy",
      className: "w-28",
      render: (p) => <span className="text-sm capitalize text-slate-400">{p.strategy.replace("_", " ")}</span>,
    },
    {
      key: "status",
      header: "Status",
      className: "w-24",
      render: (p) => {
        const variant = p.status === "healthy" ? "success" : p.status === "warning" ? "warning" : p.status === "near_stop" ? "danger" : p.status === "near_target" ? "info" : "neutral";
        return <Badge variant={variant} size="sm">{p.status.replace(/_/g, " ")}</Badge>;
      },
    },
    {
      key: "confidence",
      header: "Conf",
      className: "w-16",
      render: (p) => (
        <span className={`text-sm font-semibold tabular-nums ${getConfidenceColor(p.confidence)}`}>
          {(p.confidence * 100).toFixed(0)}%
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Portfolio Dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">Real-time portfolio overview and position management</p>
        </div>
        <span className="text-xs text-slate-400">
          Updated {formatDateTime(mockLastUpdate)}
        </span>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Total Value"
          value={formatCurrency(summary.totalValue)}
          subValue={formatPct(summary.dailyPnlPct)}
          trend={summary.dailyPnl >= 0 ? "up" : "down"}
          color="#4F46E5"
        />
        <MetricCard
          label="Cash Balance"
          value={formatCurrency(summary.cashBalance)}
          subValue={`${((summary.cashBalance / summary.totalValue) * 100).toFixed(1)}% of NAV`}
          trend="neutral"
          color="#0EA5E9"
        />
        <MetricCard
          label="Daily PnL"
          value={formatCurrency(summary.dailyPnl)}
          subValue={formatPct(summary.dailyPnlPct)}
          trend={summary.dailyPnl >= 0 ? "up" : "down"}
          color={summary.dailyPnl >= 0 ? "#10B981" : "#EF4444"}
        />
        <MetricCard
          label="Unrealized PnL"
          value={formatCurrency(summary.unrealizedPnl)}
          subValue={`${((summary.unrealizedPnl / summary.capitalDeployed) * 100).toFixed(2)}% on deployed`}
          trend={summary.unrealizedPnl >= 0 ? "up" : "down"}
          color={summary.unrealizedPnl >= 0 ? "#10B981" : "#EF4444"}
        />
        <MetricCard
          label="Gross Exposure"
          value={formatPct(summary.capitalDeployedPct)}
          subValue={formatCurrency(summary.grossExposure)}
          trend="neutral"
          color="#F59E0B"
        />
        <MetricCard
          label="Open Positions"
          value={String(summary.openPositions)}
          subValue={`${summary.pendingOrders} pending`}
          trend="neutral"
          color="#8B5CF6"
        />
      </div>

      {/* Positions Table */}
      <div>
        <div className="mb-4 flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-indigo-500" />
          <h2 className="text-xl font-bold text-slate-800">Open Positions</h2>
          <span className="ml-2 rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-500">
            {mockPositions.length}
          </span>
        </div>
        <DataTable
          columns={columns}
          data={mockPositions}
          keyExtractor={(p) => p.id}
          onRowClick={openDetail}
          expandedContent={(p) => (
            <div className="grid grid-cols-1 gap-6 text-sm md:grid-cols-3">
              <div>
                <h4 className="mb-2 font-semibold text-slate-700">Thesis</h4>
                <p className="leading-relaxed text-slate-500">{p.thesis}</p>
              </div>
              <div>
                <h4 className="mb-2 font-semibold text-slate-700">Key Reasons</h4>
                <ul className="space-y-1">
                  {p.keyReasons.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 text-slate-500">
                      <span className="mt-0.5 text-emerald-500">•</span> {r}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="mb-2 font-semibold text-slate-700">Dissent</h4>
                <p className="leading-relaxed text-slate-500">{p.dissentNotes || "No dissenting opinions"}</p>
              </div>
            </div>
          )}
        />
      </div>

      {/* Allocation */}
      <div>
        <div className="mb-4 flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-sky-500" />
          <h2 className="text-xl font-bold text-slate-800">Allocation</h2>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <AllocationPanel title="By Sector" data={mockSectorAllocation} />
          <AllocationPanel title="By Strategy" data={mockStrategyAllocation} />
          <AllocationPanel title="By Horizon" data={mockHorizonAllocation} />
          <AllocationPanel title="By Direction" data={mockDirectionAllocation} />
        </div>
      </div>

      {/* Risk Overview */}
      <div>
        <div className="mb-4 flex items-center gap-2">
          <div className="h-2.5 w-2.5 rounded-full bg-amber-500" />
          <h2 className="text-xl font-bold text-slate-800">Risk Overview</h2>
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Risk Gauges */}
          <div className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm lg:col-span-2">
            <h3 className="mb-5 text-base font-semibold text-slate-700">Portfolio Health</h3>
            <div className="space-y-5">
              <ProgressBar
                label="Drawdown"
                value={mockRiskSummary.currentDrawdown}
                max={mockRiskSummary.maxDrawdownLimit}
                formatValue={(v) => `${(v * 100).toFixed(2)}%`}
                formatMax={(v) => `${(v * 100).toFixed(0)}%`}
                warningThreshold={0.5}
                dangerThreshold={0.8}
              />
              <ProgressBar
                label="Gross Exposure"
                value={mockRiskSummary.grossExposure}
                max={mockRiskSummary.grossExposureLimit}
                formatValue={(v) => `${(v * 100).toFixed(1)}%`}
                formatMax={(v) => `${(v * 100).toFixed(0)}%`}
                warningThreshold={0.7}
                dangerThreshold={0.9}
              />
              <ProgressBar
                label="Top Position Concentration"
                value={topPositionPct}
                max={0.15}
                formatValue={(v) => `${(v * 100).toFixed(1)}%`}
                formatMax={() => "15%"}
                warningThreshold={0.6}
                dangerThreshold={0.85}
              />
              <ProgressBar
                label="Correlation Risk"
                value={mockRiskSummary.correlationRisk}
                max={1.0}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                formatMax={() => "100%"}
                warningThreshold={0.6}
                dangerThreshold={0.8}
              />
            </div>
          </div>

          {/* Winners & Losers */}
          <div className="space-y-4">
            <div className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm">
              <h3 className="mb-4 flex items-center gap-2 text-base font-semibold text-slate-700">
                <TrendingUp size={16} className="text-emerald-500" />
                Top Winners
              </h3>
              <div className="space-y-3">
                {topWinners.map((p) => (
                  <div
                    key={p.id}
                    className="flex cursor-pointer items-center justify-between rounded-xl px-3 py-2.5 transition-colors hover:bg-slate-50"
                    onClick={() => openDetail(p)}
                  >
                    <span className="text-sm font-bold text-slate-700">{p.symbol}</span>
                    <div className="text-right">
                      <span className="text-sm font-semibold text-emerald-500 tabular-nums">
                        {formatCurrency(p.unrealizedPnl)}
                      </span>
                      <span className="ml-2 text-xs text-emerald-400 tabular-nums">
                        {formatPct(p.unrealizedPnlPct)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm">
              <h3 className="mb-4 flex items-center gap-2 text-base font-semibold text-slate-700">
                <TrendingDown size={16} className="text-red-500" />
                Top Losers
              </h3>
              <div className="space-y-3">
                {topLosers.map((p) => (
                  <div
                    key={p.id}
                    className="flex cursor-pointer items-center justify-between rounded-xl px-3 py-2.5 transition-colors hover:bg-slate-50"
                    onClick={() => openDetail(p)}
                  >
                    <span className="text-sm font-bold text-slate-700">{p.symbol}</span>
                    <div className="text-right">
                      <span className="text-sm font-semibold text-red-500 tabular-nums">
                        {formatCurrency(p.unrealizedPnl)}
                      </span>
                      <span className="ml-2 text-xs text-red-400 tabular-nums">
                        {formatPct(p.unrealizedPnlPct)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <PositionDetail position={selectedPosition} open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  );
}
