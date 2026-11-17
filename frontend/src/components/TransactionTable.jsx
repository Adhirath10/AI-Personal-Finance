import { ArrowDownLeft, ArrowUpRight, MoreHorizontal } from "lucide-react";
import { shortDate, currency } from "../utils/format";

function initials(value = "") {
  return value.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

export default function TransactionTable({ rows = [], onEdit, onDelete, compact = false }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr><th>Transaction</th>{!compact && <th>Transaction ID</th>}<th>Date</th><th>Category</th><th>Type</th><th className="align-right">Amount</th><th aria-label="Actions" /></tr></thead>
        <tbody>
          {rows.map((row) => {
            const income = row.type === "Income";
            return (
              <tr key={row.id}>
                <td><div className="transaction-cell"><span className={`merchant-logo ${income ? "income" : ""}`}>{initials(row.merchant)}</span><span><strong>{row.description}</strong><small>{row.merchant}</small></span></div></td>
                {!compact && <td className="muted-cell">{row.id}</td>}
                <td className="muted-cell">{shortDate(row.date)}</td>
                <td><span className="category-chip">{row.category}</span>{row.suggested_category && <small className="category-suggestion">Model suggestion: {row.suggested_category} ({Math.round(row.category_confidence * 100)}%)</small>}</td>
                <td><span className={`type-label ${income ? "income" : "expense"}`}>{income ? <ArrowDownLeft size={13} /> : <ArrowUpRight size={13} />}{row.type}</span></td>
                <td className={`amount-cell align-right ${income ? "income-text" : ""}`}>{income ? "+" : "−"}{currency(row.amount)}</td>
                <td className="table-action">
                  {(onEdit || onDelete) && <details className="row-menu"><summary aria-label="Transaction actions"><MoreHorizontal size={17} /></summary><div className="row-menu-pop">{onEdit && <button onClick={() => onEdit(row)}>Edit transaction</button>}{onDelete && <button className="danger-action" onClick={() => onDelete(row)}>Delete transaction</button>}</div></details>}
                </td>
              </tr>
            );
          })}
          {rows.length === 0 && <tr><td className="empty-table" colSpan={compact ? 6 : 7}>No transactions found for these filters.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}
