import { useEffect, useRef, useState } from "react";
import { Bell, CalendarDays, Check, ChevronDown, Menu, Moon, Search, Sun } from "lucide-react";
import { currency } from "../utils/format";

const monthLabel = (month) => new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(
  new Date(`${month}-01T00:00:00`),
);

function PeriodPicker({ analysisPeriod, periods, selectedMonth, onSelectMonth }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const items = periods?.items || [];
  const defaultMonth = periods?.default_month;

  useEffect(() => {
    if (!open) return undefined;
    const closeOnOutside = (event) => {
      if (!containerRef.current?.contains(event.target)) setOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", closeOnOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  const choose = (month) => {
    setOpen(false);
    onSelectMonth(month === defaultMonth ? null : month);
  };
  const activeMonth = selectedMonth || defaultMonth;

  return (
    <div className="period-picker" ref={containerRef}>
      <button
        className="date-select"
        onClick={() => setOpen((value) => !value)}
        disabled={!items.length}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Choose the month to analyse"
      >
        <CalendarDays size={16} /><span>{analysisPeriod ? monthLabel(analysisPeriod) : "No period"}</span><ChevronDown size={14} />
      </button>
      {open && (
        <div className="period-menu" role="listbox" aria-label="Analysis month">
          <div className="period-menu-head">Analysis month</div>
          <div className="period-menu-list">
            {items.map((item) => {
              const selected = item.month === activeMonth;
              const incomeOnly = item.expenses === 0 && item.income > 0;
              return (
                <button
                  key={item.month}
                  role="option"
                  aria-selected={selected}
                  className={`period-option ${selected ? "selected" : ""}`}
                  onClick={() => choose(item.month)}
                >
                  <span className="period-option-main">
                    <strong>{monthLabel(item.month)}</strong>
                    <small>
                      {incomeOnly
                        ? `Income only · ${currency(item.income)}`
                        : `${currency(item.expenses)} spent · ${item.transaction_count} txns`}
                      {item.month === defaultMonth ? " · latest with expenses" : ""}
                    </small>
                  </span>
                  {selected && <Check size={15} />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Navbar({
  onMenu,
  onSearch,
  onTheme,
  theme,
  mode,
  analysisPeriod,
  periods,
  selectedMonth,
  onSelectMonth,
  onNotifications,
  onProfile,
}) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-btn menu-trigger" onClick={onMenu} aria-label="Open navigation"><Menu size={20} /></button>
        <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>Personal finance</strong></div>
      </div>
      <div className="topbar-actions">
        <label className="global-search">
          <Search size={16} />
          <input type="search" placeholder="Search anything..." onChange={(event) => onSearch(event.target.value)} />
          <kbd>⌘ K</kbd>
        </label>
        <span className={`connection-pill ${mode}`}>
          <span className="connection-dot" />{mode === "live" ? "Live API" : mode === "unavailable" ? "API unavailable" : "Connecting"}
        </span>
        <button className="icon-btn" aria-label="Notifications" title="Notifications" onClick={onNotifications}><Bell size={18} /></button>
        <button className="icon-btn theme-toggle" onClick={onTheme} aria-label="Toggle theme" title="Toggle theme">{theme === "light" ? <Moon size={18} /> : <Sun size={18} />}</button>
        <PeriodPicker analysisPeriod={analysisPeriod} periods={periods} selectedMonth={selectedMonth} onSelectMonth={onSelectMonth} />
        <button className="profile-button" aria-label="Open profile" onClick={onProfile}><span className="profile-avatar">PF</span><ChevronDown size={14} /></button>
      </div>
    </header>
  );
}
