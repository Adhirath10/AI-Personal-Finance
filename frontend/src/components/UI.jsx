import {
  ArrowDownRight,
  ArrowUpRight,
  ChevronRight,
  CircleHelp,
  Info,
  Plus,
} from "lucide-react";
import { currency, number, percent } from "../utils/format";

export function PageHeading({ eyebrow, title, description, actions }) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="heading-actions">{actions}</div>}
    </div>
  );
}

export function SummaryCard({ label, value, change, icon: Icon, tone, helper, prefix, format = "currency" }) {
  const positive = Number(change) >= 0;
  const ChangeIcon = positive ? ArrowUpRight : ArrowDownRight;
  const displayValue = (() => {
    if (value == null || value === "—") return value;
    if (typeof value === "number") {
      if (format === "number") return number(value);
      if (format === "percent") return percent(value);
      return currency(value, true);
    }
    return value;
  })();

  return (
    <article className="summary-card">
      <div className="summary-top"><span>{label}</span><div className={`summary-icon ${tone}`}><Icon size={18} /></div></div>
      <div className="summary-value">{prefix}{displayValue}</div>
      <div className="summary-foot">
        {change !== undefined && <span className={`change-pill ${positive ? "positive" : "negative"}`}><ChangeIcon size={13} />{Math.abs(Number(change)).toFixed(1)}%</span>}
        {helper && <span className="summary-helper">{helper}</span>}
      </div>
    </article>
  );
}

export function ChartCard({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`panel chart-panel ${className}`}>
      <div className="panel-heading">
        <div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function StatusBadge({ children, tone = "neutral" }) {
  return <span className={`status-badge ${tone}`}><i />{children}</span>;
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Info size={20} /></div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function LoadingSkeleton({ rows = 3 }) {
  return <div className="skeleton-list" aria-label="Loading"><div className="skeleton skeleton-title" />{Array.from({ length: rows }, (_, index) => <div className="skeleton skeleton-row" key={index} />)}</div>;
}

export function Button({ children, variant = "primary", icon: Icon, onClick, type = "button", disabled = false, ...props }) {
  return <button type={type} className={`button button-${variant}`} onClick={onClick} disabled={disabled} {...props}>{Icon && <Icon size={16} />}{children}</button>;
}

export function SectionLink({ children, to }) {
  return <a className="section-link" href={to}>{children}<ChevronRight size={15} /></a>;
}

export function BudgetProgress({ budget, compact = false }) {
  const ratio = budget.amount > 0 ? Math.round((budget.spent / budget.amount) * 100) : 0;
  const tone = ratio >= 100 ? "danger" : ratio >= 80 ? "warning" : "success";
  return (
    <div className={`budget-progress ${compact ? "compact" : ""}`}>
      <div className="budget-meta"><span>{budget.category}</span><strong>{currency(budget.spent)} <small>of {currency(budget.amount)}</small></strong></div>
      <div className={`progress-track ${tone}`}><span style={{ width: `${Math.min(ratio, 100)}%` }} /></div>
      {!compact && <div className="budget-bottom"><span>{percent(ratio)} used</span><span className={ratio >= 100 ? "text-danger" : ""}>{ratio >= 100 ? `${currency(budget.spent - budget.amount)} over` : `${currency(budget.amount - budget.spent)} left`}</span></div>}
    </div>
  );
}

export function EmptyAction({ onClick, children = "Add new" }) {
  return <Button icon={Plus} onClick={onClick}>{children}</Button>;
}

export function TooltipNote({ children }) {
  return <span className="tooltip-note" title={children}><CircleHelp size={14} /></span>;
}
