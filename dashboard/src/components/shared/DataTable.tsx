"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => React.ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (row: T) => string;
  onRowClick?: (row: T) => void;
  expandedContent?: (row: T) => React.ReactNode;
  className?: string;
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onRowClick,
  expandedContent,
  className = "",
}: DataTableProps<T>) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const toggleExpand = (key: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div className={`overflow-x-auto rounded-2xl border border-slate-100 bg-white shadow-sm ${className}`}>
      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50/80">
            {expandedContent && <th className="w-10 px-4 py-3.5" />}
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-4 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-400 ${col.className || ""}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const key = keyExtractor(row);
            const isExpanded = expandedRows.has(key);
            return (
              <tr key={key}>
                <td colSpan={columns.length + (expandedContent ? 1 : 0)} className="p-0">
                  <div
                    className={`flex cursor-pointer items-center border-b border-slate-50 transition-colors hover:bg-slate-50/60 ${
                      isExpanded ? "bg-slate-50/40" : ""
                    }`}
                    onClick={() => {
                      if (expandedContent) toggleExpand(key);
                      onRowClick?.(row);
                    }}
                  >
                    {expandedContent && (
                      <div className="flex w-10 shrink-0 items-center justify-center px-4 py-3.5">
                        {isExpanded ? (
                          <ChevronDown size={14} className="text-slate-300" />
                        ) : (
                          <ChevronRight size={14} className="text-slate-300" />
                        )}
                      </div>
                    )}
                    {columns.map((col) => (
                      <div key={col.key} className={`shrink-0 px-4 py-3.5 ${col.className || ""}`}>
                        {col.render(row)}
                      </div>
                    ))}
                  </div>
                  {expandedContent && isExpanded && (
                    <div className="border-b border-slate-100 bg-slate-50/60 px-12 py-5">
                      {expandedContent(row)}
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
