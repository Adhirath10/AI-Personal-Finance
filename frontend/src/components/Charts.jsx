import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { currency } from "../utils/format";

const axis = { fill: "#919caf", fontSize: 11 };

export function IncomeExpenseChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} barGap={7} margin={{ top: 8, right: 4, left: -12, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="#edf0f5" />
        <XAxis dataKey="month" axisLine={false} tickLine={false} tick={axis} dy={9} />
        <YAxis axisLine={false} tickLine={false} tick={axis} tickFormatter={(value) => `₹${Math.round(value / 1000)}k`} />
        <Tooltip formatter={(value) => currency(value)} contentStyle={{ borderRadius: 12, border: "1px solid #e8ebf2", boxShadow: "0 10px 24px #17264a12", fontSize: 12 }} />
        <Bar dataKey="income" name="Income" fill="#5278f5" radius={[5, 5, 0, 0]} maxBarSize={24} />
        <Bar dataKey="expenses" name="Expenses" fill="#c6d2fb" radius={[5, 5, 0, 0]} maxBarSize={24} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ExpenseDonut({ data = [] }) {
  return (
    <div className="donut-wrap">
      <ResponsiveContainer width="100%" height={206}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={66} outerRadius={88} paddingAngle={3} stroke="none">
            {data.map((item) => <Cell key={item.name} fill={item.color || "#5378f6"} />)}
          </Pie>
          <Tooltip formatter={(value) => currency(value)} contentStyle={{ borderRadius: 12, border: "1px solid #e8ebf2", fontSize: 12 }} />
        </PieChart>
      </ResponsiveContainer>
      <div className="donut-center"><strong>{currency(data.reduce((sum, item) => sum + Number(item.value || 0), 0), true)}</strong><small>Total spend</small></div>
    </div>
  );
}

export function SpendingLineChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 12, right: 8, left: -12, bottom: 0 }}>
        <defs><linearGradient id="spendingFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5578f5" stopOpacity={0.19} /><stop offset="95%" stopColor="#5578f5" stopOpacity={0} /></linearGradient></defs>
        <CartesianGrid vertical={false} stroke="#edf0f5" />
        <XAxis dataKey="month" axisLine={false} tickLine={false} tick={axis} dy={9} />
        <YAxis axisLine={false} tickLine={false} tick={axis} tickFormatter={(value) => `₹${Math.round(value / 1000)}k`} />
        <Tooltip formatter={(value) => currency(value)} contentStyle={{ borderRadius: 12, border: "1px solid #e8ebf2", boxShadow: "0 10px 24px #17264a12", fontSize: 12 }} />
        <Area type="monotone" dataKey="expenses" name="Expenses" stroke="#5578f5" strokeWidth={2.5} fill="url(#spendingFill)" activeDot={{ r: 5, strokeWidth: 0 }} />
        {data.some((item) => item.forecast) && <Area type="monotone" dataKey="forecast" name="Forecast" stroke="#f0a94a" strokeWidth={2} strokeDasharray="5 5" fill="transparent" connectNulls />}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function FeatureImportanceChart({ data = [] }) {
  const rows = [...data].sort((a, b) => Math.abs(b.value ?? b.importance ?? b.mean_absolute_shap ?? 0) - Math.abs(a.value ?? a.importance ?? a.mean_absolute_shap ?? 0)).slice(0, 8);
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, rows.length * 38)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 20, left: 5, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="#edf0f5" />
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="feature" width={132} axisLine={false} tickLine={false} tick={{ ...axis, fill: "#59657a" }} />
        <Tooltip formatter={(value) => Number(value).toFixed(4)} />
        <Bar dataKey={(item) => Math.abs(item.value ?? item.importance ?? item.mean_absolute_shap ?? item.shap_value ?? 0)} name="Mean |SHAP|" fill="#5378f6" radius={[0, 5, 5, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}
