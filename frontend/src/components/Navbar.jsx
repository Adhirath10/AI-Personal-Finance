import { useEffect, useRef, useState } from "react";
import { Bell, BellOff, CalendarDays, Check, ChevronDown, ChevronRight, Menu, Moon, Search, Sun } from "lucide-react";
import { currency } from "../utils/format";
import { Avatar } from "./UI";

const monthLabel = (month) => new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(
  new Date(`${month}-01T00:00:00`),
);

// Close a popover on outside click or Escape.
function useDismiss(open, setOpen) {
  const containerRef = useRef(null);
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
  }, [open, setOpen]);
  return containerRef;
}

function PeriodPicker({ analysisPeriod, periods, selectedMonth, onSelectMonth }) {
  const [open, setOpen] = useState(false);
  const containerRef = useDismiss(open, setOpen);
  const items = periods?.items || [];
  const defaultMonth = periods?.default_month;

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
                      {item.month === defaultMonth ? " · latest" : ""}
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

function NotificationsMenu({ notifications, enabled, onOpenItem, onOpenSettings }) {
  const [open, setOpen] = useState(false);
  const [seen, setSeen] = useState(() => new Set());
  const containerRef = useDismiss(open, setOpen);
  const unread = enabled ? notifications.filter((item) => !seen.has(item.id)).length : 0;

  const toggle = () => {
    setOpen((value) => !value);
    setSeen(new Set(notifications.map((item) => item.id)));
  };
  const openItem = (item) => {
    setOpen(false);
    onOpenItem(item.to);
  };

  return (
    <div className="notifications" ref={containerRef}>
      <button className="icon-btn" aria-label={`Notifications${unread ? ` (${unread} new)` : ""}`} title="Notifications" aria-expanded={open} onClick={toggle}>
        {enabled ? <Bell size={18} /> : <BellOff size={18} />}
        {unread > 0 && <span className="notification-badge">{unread > 9 ? "9+" : unread}</span>}
      </button>
      {open && (
        <div className="notification-menu" role="dialog" aria-label="Notifications">
          <div className="period-menu-head">Notifications</div>
          {!enabled ? (
            <div className="notification-empty">
              <p>Notifications are turned off.</p>
              <button className="text-button" onClick={() => { setOpen(false); onOpenSettings(); }}>Turn on in Settings <ChevronRight size={13} /></button>
            </div>
          ) : notifications.length ? (
            <div className="notification-list">
              {notifications.map((item) => (
                <button key={item.id} className="notification-item" onClick={() => openItem(item)}>
                  <i className={`notification-dot-tone ${item.tone}`} />
                  <span><strong>{item.title}</strong><small>{item.body}</small></span>
                  <ChevronRight size={14} />
                </button>
              ))}
            </div>
          ) : (
            <div className="notification-empty"><Check size={18} /><p>You're all caught up.</p></div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Navbar({
  onMenu,
  onSearch,
  onSearchSubmit,
  onTheme,
  theme,
  mode,
  analysisPeriod,
  periods,
  selectedMonth,
  onSelectMonth,
  notifications = [],
  notificationsEnabled = true,
  onNavigate,
  onProfile,
  profile = {},
}) {
  const searchRef = useRef(null);
  useEffect(() => {
    const focusSearch = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
      }
    };
    document.addEventListener("keydown", focusSearch);
    return () => document.removeEventListener("keydown", focusSearch);
  }, []);
  const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || "");

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-btn menu-trigger" onClick={onMenu} aria-label="Open navigation"><Menu size={20} /></button>
        <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>Personal finance</strong></div>
      </div>
      <div className="topbar-actions">
        <form className="global-search" role="search" onSubmit={(event) => { event.preventDefault(); onSearchSubmit(searchRef.current?.value || ""); }}>
          <Search size={16} />
          <input ref={searchRef} type="search" placeholder="Search transactions..." aria-label="Search transactions" onChange={(event) => onSearch(event.target.value)} />
          <kbd>{isMac ? "⌘ K" : "Ctrl K"}</kbd>
        </form>
        <span className={`connection-pill ${mode}`}>
          <span className="connection-dot" />{mode === "live" ? "Live API" : mode === "unavailable" ? "API unavailable" : "Connecting"}
        </span>
        <NotificationsMenu notifications={notifications} enabled={notificationsEnabled} onOpenItem={onNavigate} onOpenSettings={onProfile} />
        <button className="icon-btn theme-toggle" onClick={onTheme} aria-label="Toggle theme" title="Toggle theme">{theme === "light" ? <Moon size={18} /> : <Sun size={18} />}</button>
        <PeriodPicker analysisPeriod={analysisPeriod} periods={periods} selectedMonth={selectedMonth} onSelectMonth={onSelectMonth} />
        <button className="profile-button" aria-label="Open profile settings" title={profile.name || "Profile settings"} onClick={onProfile}><Avatar name={profile.name} photo={profile.photo} /><ChevronDown size={14} /></button>
      </div>
    </header>
  );
}
