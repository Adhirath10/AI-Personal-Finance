import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  Briefcase, CalendarClock, Car, CheckCircle2, CircleAlert, GraduationCap, Heart, House, Landmark,
  Lightbulb, PiggyBank, Plane, Scissors, ShieldCheck, Smartphone, Target, TrendingDown,
} from "lucide-react";
import { Button, StatusBadge } from "./UI";
import { currency } from "../utils/format";
import { GOAL_PRESETS, GOAL_TIPS, planGoal } from "../utils/goalPlanner";

const PRESET_ICONS = { car: Car, home: House, vacation: Plane, emergency: ShieldCheck, wedding: Heart, education: GraduationCap, gadget: Smartphone, custom: Target };
const DEFAULT_FINANCE = { enabled: false, downPct: 20, interest: 9, years: 5 };

const compactRupees = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", notation: "compact", maximumFractionDigits: 1 });
const monthYear = (date) => new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(date);
const addMonths = (count) => {
  const date = new Date();
  date.setMonth(date.getMonth() + count);
  return date;
};
const duration = (months) => {
  if (!Number.isFinite(months)) return "never at this rate";
  const years = Math.floor(months / 12);
  const rest = months % 12;
  return [years ? `${years} yr${years > 1 ? "s" : ""}` : "", rest ? `${rest} mo` : ""].filter(Boolean).join(" ") || "now";
};

function NumberField({ label, value, onChange, prefix, suffix, min = 0, step = 1000 }) {
  return (
    <label className="goal-field">{label}
      <span className="goal-input">{prefix && <i>{prefix}</i>}<input type="number" min={min} step={step} value={value} onChange={(event) => onChange(Math.max(min, Number(event.target.value) || 0))} />{suffix && <i>{suffix}</i>}</span>
    </label>
  );
}

export default function GoalPlanner({ income, expenses, mode, onApplyCuts, onSaveGoal }) {
  const monthlyExpenses = Object.values(expenses).reduce((sum, value) => sum + (Number(value) || 0), 0);
  const emergencyTarget = Math.max(50000, Math.round((monthlyExpenses * 6) / 10000) * 10000);
  const [presetKey, setPresetKey] = useState("car");
  const preset = GOAL_PRESETS.find((item) => item.key === presetKey);
  const [name, setName] = useState(preset.name);
  const [target, setTarget] = useState(preset.target);
  const [saved, setSaved] = useState(0);
  const [months, setMonths] = useState(preset.months);
  const [annualReturn, setAnnualReturn] = useState(6);
  const [finance, setFinance] = useState(preset.finance || DEFAULT_FINANCE);
  const [saving, setSaving] = useState(false);

  const choosePreset = (key) => {
    const next = GOAL_PRESETS.find((item) => item.key === key);
    setPresetKey(key);
    setName(next.name);
    setTarget(next.target ?? emergencyTarget);
    setMonths(next.months);
    setFinance(next.finance || DEFAULT_FINANCE);
  };

  const plan = useMemo(
    () => planGoal({ target, saved, months, annualReturn, income, expenses, finance, preset: presetKey }),
    [target, saved, months, annualReturn, income, expenses, finance, presetKey],
  );
  const goalName = finance.enabled ? `${name} down payment` : name;
  const tips = [...(GOAL_TIPS[presetKey] || []), ...GOAL_TIPS.default];
  const readyBy = plan.status === "stretch" ? plan.monthsWithMaxCuts : Math.round(months);
  const verdict = plan.needed === 0
    ? { tone: "success", icon: CheckCircle2, title: "You've already saved enough", text: `Your ${currency(saved)} covers the ${currency(plan.savingsTarget)} you need.` }
    : plan.status === "on_track" && plan.loan && !plan.loan.affordable
      ? { tone: "warning", icon: CircleAlert, title: "The down payment is easy; the EMI is high", text: `Save ${currency(plan.required)} a month for the ${currency(plan.savingsTarget)} down payment. But ${presetKey === "car" ? "running the car and paying the EMI" : "the EMI"} would take ${Math.round((plan.loan.share || 0) * 100)}% of your income — see the loan section below.` }
    : plan.status === "on_track"
      ? { tone: "success", icon: CheckCircle2, title: "You can afford this", text: `Save ${currency(plan.required)} a month. You already have ${currency(plan.surplus)} left over each month, so ${currency(plan.surplus - plan.required)} stays free.` }
      : plan.status === "with_cuts"
        ? { tone: "warning", icon: Scissors, title: "Possible with a few cuts", text: `You need ${currency(plan.required)} a month but have ${currency(Math.max(plan.surplus, 0))} left over. Trimming ${currency(plan.cutTotal)} a month, as below, closes the gap.` }
        : { tone: "danger", icon: CircleAlert, title: "This is a stretch", text: `Even after the suggested cuts you would be ${currency(plan.remaining)} a month short. See the other options below.` };
  const VerdictIcon = verdict.icon;

  const save = async () => {
    setSaving(true);
    try {
      const date = plan.goalDate.toISOString().slice(0, 10);
      await onSaveGoal({ name: goalName, target: Math.round(plan.savingsTarget), current: Math.round(saved), date });
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="panel goal-planner">
      <div className="panel-heading"><div><h2><PiggyBank size={18} />Goal planner</h2><p>Planning a big purchase? See how much to save each month, what to cut and other ways to get there, using your numbers above.</p></div></div>
      <div className="goal-presets" role="tablist" aria-label="Goal type">
        {GOAL_PRESETS.map((item) => {
          const Icon = PRESET_ICONS[item.key];
          return <button key={item.key} role="tab" aria-selected={presetKey === item.key} className={`goal-preset ${presetKey === item.key ? "active" : ""}`} onClick={() => choosePreset(item.key)}><Icon size={16} />{item.label}</button>;
        })}
      </div>

      <div className="goal-planner-grid">
        <div className="goal-inputs">
          <label className="goal-field">Goal name<span className="goal-input"><input value={name} onChange={(event) => setName(event.target.value)} /></span></label>
          <NumberField label={finance.enabled ? "Price" : "Target amount"} value={target} onChange={setTarget} prefix="₹" step={10000} />
          <NumberField label="Already saved for it" value={saved} onChange={setSaved} prefix="₹" step={5000} />
          <label className="goal-field">Time to goal <strong>{months} months · by {monthYear(addMonths(months))}</strong>
            <input type="range" min="3" max="120" step="1" value={months} onChange={(event) => setMonths(Number(event.target.value))} />
          </label>
          <NumberField label="Expected return on savings" value={annualReturn} onChange={(value) => setAnnualReturn(Math.min(value, 15))} suffix="% a year" step={0.5} />
          <div className="goal-finance">
            <div className="setting-row"><div><strong>Take a loan for the rest</strong><span>Save only the down payment, pay the rest as EMI.</span></div><button type="button" role="switch" aria-checked={finance.enabled} aria-label="Take a loan" className={`toggle ${finance.enabled ? "on" : ""}`} onClick={() => setFinance((current) => ({ ...current, enabled: !current.enabled }))}><i /></button></div>
            {finance.enabled && <div className="goal-finance-fields">
              <label className="goal-field">Down payment <strong>{finance.downPct}% · {currency(plan.savingsTarget)}</strong><input type="range" min="10" max="90" step="5" value={finance.downPct} onChange={(event) => setFinance((current) => ({ ...current, downPct: Number(event.target.value) }))} /></label>
              <NumberField label="Loan interest" value={finance.interest} onChange={(value) => setFinance((current) => ({ ...current, interest: Math.min(value, 30) }))} suffix="% a year" step={0.25} />
              <NumberField label="Loan tenure" value={finance.years} onChange={(value) => setFinance((current) => ({ ...current, years: Math.min(Math.max(value, 1), 30) }))} suffix="years" step={1} min={1} />
            </div>}
          </div>
        </div>

        <div className="goal-results">
          <div className={`goal-verdict ${verdict.tone}`}><VerdictIcon size={20} /><div><strong>{verdict.title}</strong><p>{verdict.text}</p></div></div>
          <div className="goal-metrics">
            <div className="goal-metric"><span>Save every month</span><strong>{currency(plan.required)}</strong><small>for {finance.enabled ? "the down payment" : "the goal"} of {currency(plan.savingsTarget)}</small></div>
            <div className="goal-metric"><span>You save now</span><strong className={plan.surplus < 0 ? "expense-text" : ""}>{currency(plan.surplus)}</strong><small>income minus expenses above</small></div>
            <div className="goal-metric"><span>{plan.gap > 0 ? "Monthly gap" : "Room to spare"}</span><strong className={plan.gap > 0 ? "expense-text" : "income-text"}>{currency(Math.abs(plan.gap))}</strong><small>{plan.gap > 0 ? "to find each month" : "after this saving"}</small></div>
            <div className="goal-metric"><span>Ready by</span><strong>{Number.isFinite(readyBy) ? monthYear(addMonths(readyBy)) : "—"}</strong><small>{plan.status === "stretch" ? `with all cuts: ${duration(plan.monthsWithMaxCuts)}` : `in ${duration(Math.round(months))}`}</small></div>
          </div>

          {plan.cuts.length > 0 && <div className="goal-block">
            <h3><Scissors size={16} />What to cut back</h3>
            <div className="table-scroll"><table className="data-table goal-cuts"><thead><tr><th>Category</th><th className="align-right">Now</th><th className="align-right">Plan</th><th className="align-right">Saves</th><th>How</th></tr></thead><tbody>
              {plan.cuts.map((cut) => <tr key={cut.category}><td><strong>{cut.category}</strong></td><td className="align-right">{currency(cut.current)}</td><td className="align-right">{currency(cut.suggested)}</td><td className="align-right income-text">{currency(cut.saving)}</td><td className="muted-cell">{cut.tip}</td></tr>)}
            </tbody></table></div>
            <div className="goal-block-footer"><span>Total: <strong>{currency(plan.cutTotal)}</strong> a month. Rent, EMIs, insurance, health and education are never cut.</span><Button variant="outline" icon={TrendingDown} onClick={() => onApplyCuts(Object.fromEntries(plan.cuts.map((cut) => [cut.category, cut.suggested])))}>Apply to simulator</Button></div>
          </div>}

          <div className="goal-block">
            <h3><Lightbulb size={16} />Other ways to get there</h3>
            <ul className="goal-options">
              {plan.needed > 0 && Number.isFinite(plan.monthsAtCurrent) && <li><CalendarClock size={16} /><span>{plan.monthsAtCurrent <= months ? `At your current savings you'd reach it in ${duration(plan.monthsAtCurrent)} — no cuts needed.` : `Without any cuts, give it ${duration(plan.monthsAtCurrent)} (by ${monthYear(addMonths(plan.monthsAtCurrent))}) instead of ${duration(Math.round(months))}.`}</span></li>}
              {plan.status === "stretch" && Number.isFinite(plan.monthsWithMaxCuts) && <li><Scissors size={16} /><span>With all the suggested cuts you'd get there in {duration(plan.monthsWithMaxCuts)} (by {monthYear(addMonths(plan.monthsWithMaxCuts))}).</span></li>}
              {plan.remaining > 0 && <li><Briefcase size={16} /><span>Earn {currency(plan.remaining)} more a month (freelancing, overtime, a side project) to keep the {duration(Math.round(months))} timeline.</span></li>}
              {plan.status === "stretch" && <li><Target size={16} /><span>In {duration(Math.round(months))} you can realistically save about {currency(plan.affordableSavings)}{finance.enabled ? ` — enough down payment for a price of about ${currency(plan.affordableSavings / (finance.downPct / 100))}` : ""}.</span></li>}
              {plan.needed > 0 && <li><PiggyBank size={16} /><span>Put bonuses or tax refunds towards it: every {currency(50000)} brings the goal about {Math.max(1, Math.round(50000 / Math.max(plan.plannedMonthly || plan.required, 1)))} month(s) closer.</span></li>}
              {!finance.enabled && plan.price >= 300000 && <li><Landmark size={16} /><span>Or save only a down payment and borrow the rest: switch on “Take a loan” to see the EMI.</span></li>}
            </ul>
          </div>

          {plan.loan && <div className={`goal-block goal-loan ${plan.loan.affordable ? "ok" : "warn"}`}>
            <h3><Landmark size={16} />Loan after the down payment</h3>
            <div className="goal-loan-grid">
              <div><span>Loan amount</span><strong>{currency(plan.loan.principal)}</strong></div>
              <div><span>EMI</span><strong>{currency(plan.loan.emi)}/mo</strong><small>for {finance.years} yrs at {finance.interest}%</small></div>
              <div><span>Total interest</span><strong>{currency(plan.loan.totalInterest)}</strong></div>
              {plan.loan.runningCost > 0 && <div><span>Fuel, insurance, service</span><strong>≈ {currency(plan.loan.runningCost)}/mo</strong></div>}
            </div>
            <p>
              <StatusBadge tone={plan.loan.affordable ? "success" : "warning"}>{plan.loan.affordable ? "Within guideline" : "Above guideline"}</StatusBadge>{" "}
              {presetKey === "car" ? "Car costs" : "The EMI"} would take {Math.round((plan.loan.share || 0) * 100)}% of your income (guideline: under {Math.round(plan.loan.guideline * 100)}%).
              {!plan.loan.affordable && plan.loan.affordablePrice ? ` A price up to about ${currency(plan.loan.affordablePrice)} keeps it within the guideline, or choose a bigger down payment or a longer tenure.` : ""}
              {" "}After buying, you'd have {currency(plan.loan.budgetAfter)} a month left over{plan.cutTotal ? " (with the cuts above)" : ""}.
            </p>
          </div>}

          <div className="goal-block">
            <h3><TrendingDown size={16} style={{ transform: "scaleY(-1)" }} />Savings projection</h3>
            <div className="goal-chart">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={plan.projection} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <defs><linearGradient id="goalFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5378f6" stopOpacity={0.35} /><stop offset="100%" stopColor="#5378f6" stopOpacity={0.02} /></linearGradient></defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" vertical={false} />
                  <XAxis dataKey="month" tickFormatter={(value) => monthYear(addMonths(value))} tick={{ fontSize: 10, fill: "var(--muted)" }} axisLine={false} tickLine={false} minTickGap={24} />
                  <YAxis domain={[0, (dataMax) => Math.max(dataMax, plan.savingsTarget) * 1.08]} tickFormatter={(value) => compactRupees.format(value)} tick={{ fontSize: 10, fill: "var(--muted)" }} axisLine={false} tickLine={false} width={64} />
                  <Tooltip formatter={(value) => currency(value)} labelFormatter={(value) => monthYear(addMonths(value))} contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 10, fontSize: 11 }} />
                  <ReferenceLine y={plan.savingsTarget} stroke="#28a77e" strokeDasharray="5 4" label={{ value: "Target", position: "insideTopRight", fontSize: 10, fill: "#28a77e" }} />
                  <Area type="monotone" dataKey="saved" name="Saved" stroke="#5378f6" strokeWidth={2.5} fill="url(#goalFill)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <small className="goal-chart-note">Saving {currency(plan.plannedMonthly)} a month{annualReturn ? ` at ${annualReturn}% a year` : ""}.</small>
          </div>

          <div className="goal-block">
            <h3><ShieldCheck size={16} />Good habits for this goal</h3>
            <ul className="goal-tips">{tips.map((tip) => <li key={tip}>{tip}</li>)}</ul>
          </div>

          <div className="goal-save"><span>Track it on the Financial goals page.</span><Button icon={Target} disabled={mode !== "live" || saving || plan.savingsTarget <= 0} onClick={save}>{saving ? "Saving..." : `Save “${goalName}” as a goal`}</Button></div>
        </div>
      </div>
    </section>
  );
}
