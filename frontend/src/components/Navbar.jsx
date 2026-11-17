import { Bell, CalendarDays, ChevronDown, Menu, Moon, Search, Sun } from "lucide-react";

export default function Navbar({
  onMenu,
  onSearch,
  onTheme,
  theme,
  mode,
  onDate,
  analysisPeriod,
  onNotifications,
  onProfile,
}) {
  const periodLabel = analysisPeriod
    ? new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(
      new Date(`${analysisPeriod}-01T00:00:00`),
    )
    : "No period";
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
        <button className="date-select" onClick={onDate} title="Open transaction date filters"><CalendarDays size={16} /><span>{periodLabel}</span><ChevronDown size={14} /></button>
        <button className="profile-button" aria-label="Open profile" onClick={onProfile}><span className="profile-avatar">PF</span><ChevronDown size={14} /></button>
      </div>
    </header>
  );
}
