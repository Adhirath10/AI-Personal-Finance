import { NavLink } from "react-router-dom";
import {
  ArrowDownUp,
  BadgeIndianRupee,
  BarChart3,
  BellRing,
  ChevronLeft,
  CircleHelp,
  Crosshair,
  Goal,
  LayoutDashboard,
  Lightbulb,
  ListChecks,
  PanelLeftClose,
  Settings,
  ShieldAlert,
  Sparkles,
  WalletCards,
  X,
} from "lucide-react";
import { Avatar } from "./UI";

const navigation = [
  { label: "Overview", to: "/", icon: LayoutDashboard },
  { label: "Transactions", to: "/transactions", icon: ArrowDownUp },
  { label: "AI insights", to: "/insights", icon: Sparkles, section: "YOUR FINANCES" },
  { label: "Forecasting", to: "/forecasting", icon: BarChart3 },
  { label: "Anomaly detection", to: "/anomalies", icon: ShieldAlert },
  { label: "Recommendations", to: "/recommendations", icon: Lightbulb },
  { label: "Budgets", to: "/budgets", icon: WalletCards, section: "PLANNING" },
  { label: "Financial goals", to: "/goals", icon: Goal },
  { label: "What-if simulator", to: "/simulator", icon: Crosshair },
];

export default function Sidebar({ open, onClose, collapsed, onCollapse, recommendationCount = 0, profile = {}, onWorkspace, onHelp }) {
  return (
    <>
      <div className={`sidebar-scrim ${open ? "visible" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${open ? "sidebar-open" : ""} ${collapsed ? "sidebar-collapsed" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-mark"><BadgeIndianRupee size={21} strokeWidth={2.2} /></div>
          {!collapsed && <span>Fin<span className="brand-ai">AI</span></span>}
          <button className="icon-btn sidebar-close mobile-only" aria-label="Close navigation" onClick={onClose}><X size={18} /></button>
        </div>
        <button className="workspace-switch" title="Open profile settings" onClick={() => { onClose(); onWorkspace(); }}>
          <Avatar name={profile.name} photo={profile.photo} />
          {!collapsed && <span className="workspace-name">{profile.name || "Personal workspace"}<small>{profile.email || "Profile & preferences"}</small></span>}
          {!collapsed && <ChevronLeft size={15} className="workspace-chevron" />}
        </button>
        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.to}>
                {item.section && !collapsed && <p className="nav-section">{item.section}</p>}
                <NavLink
                  to={item.to}
                  end={item.to === "/"}
                  onClick={onClose}
                  title={collapsed ? item.label : undefined}
                  className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
                >
                  <Icon size={18} strokeWidth={1.8} />
                  {!collapsed && <span>{item.label}</span>}
                  {!collapsed && item.to === "/recommendations" && recommendationCount > 0 && <span className="nav-count">{recommendationCount}</span>}
                </NavLink>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <button className="sidebar-help" onClick={() => { onClose(); onHelp(); }} title="Open the FinAI guide">
            {!collapsed && <><div className="help-icon"><CircleHelp size={17} /></div><div><strong>Need a hand?</strong><small>Open the FinAI guide</small></div></>}
            {collapsed && <CircleHelp size={18} />}
          </button>
          <NavLink to="/settings" onClick={onClose} title={collapsed ? "Settings" : undefined} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Settings size={18} strokeWidth={1.8} />{!collapsed && <span>Settings</span>}
          </NavLink>
          <div className="sidebar-footer">
            <button className="collapse-btn" onClick={onCollapse} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
              {collapsed ? <ChevronLeft size={17} /> : <PanelLeftClose size={17} />}
              {!collapsed && <span>Collapse menu</span>}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
