import { ArrowDownUp, BarChart3, CalendarDays, Crosshair, Goal, LayoutDashboard, Lightbulb, ShieldAlert, Sparkles, WalletCards } from "lucide-react";
import Modal from "./Modal";

const sections = [
  { icon: LayoutDashboard, to: "/", title: "Overview", text: "Income, expenses, savings and budget use for the selected month, with cash-flow and category charts." },
  { icon: CalendarDays, title: "Month picker", text: "The date button at the top chooses which month every page analyses. It starts on your latest month with expenses." },
  { icon: ArrowDownUp, to: "/transactions", title: "Transactions", text: "Search, filter, add, edit and delete transactions. Auto categorize suggests categories with the trained model." },
  { icon: Sparkles, to: "/insights", title: "AI insights", text: "Compares each category's spending with the previous three recorded months." },
  { icon: BarChart3, to: "/forecasting", title: "Forecasting", text: "A Random Forest model predicts next month's total expenses; the page also shows what pushed the prediction up or down." },
  { icon: ShieldAlert, to: "/anomalies", title: "Anomaly detection", text: "An Isolation Forest flags transactions outside your usual pattern, most unusual first. Flags are for review, not proof of fraud." },
  { icon: Lightbulb, to: "/recommendations", title: "Recommendations", text: "Actions derived from budgets, trends, forecasts, anomalies and goals. Each card links to the page it is about." },
  { icon: WalletCards, to: "/budgets", title: "Budgets", text: "Monthly category limits compared with what was actually spent." },
  { icon: Goal, to: "/goals", title: "Financial goals", text: "Savings targets with progress and deadlines." },
  { icon: Crosshair, to: "/simulator", title: "What-if simulator", text: "Change income or category spending to see the effect on monthly savings." },
];

export default function HelpGuide({ onClose, onNavigate }) {
  return (
    <Modal title="FinAI guide" subtitle="What each part of the dashboard does. Select a page to open it." onClose={onClose}>
      <div className="help-grid">
        {sections.map(({ icon: Icon, to, title, text }) => {
          const content = <><span className="help-grid-icon"><Icon size={17} /></span><span><strong>{title}</strong><small>{text}</small></span></>;
          return to
            ? <button key={title} className="help-grid-item" onClick={() => { onClose(); onNavigate(to); }}>{content}</button>
            : <div key={title} className="help-grid-item">{content}</div>;
        })}
      </div>
      <p className="help-footnote">Tip: press {typeof navigator !== "undefined" && /Mac/.test(navigator.platform || "") ? "⌘" : "Ctrl"} + K to search transactions from anywhere.</p>
    </Modal>
  );
}
