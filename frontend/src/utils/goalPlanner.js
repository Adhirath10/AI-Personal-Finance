// Savings-goal planning: how much to save each month, which expenses to trim,
// and what else could get the user there. Pure functions so they are easy to test.

export const GOAL_PRESETS = [
  { key: "car", label: "Buy a car", name: "New car", target: 800000, months: 24, finance: { enabled: true, downPct: 20, interest: 9, years: 5 }, emiGuideline: 0.15 },
  { key: "home", label: "Buy a home", name: "Home", target: 5000000, months: 48, finance: { enabled: true, downPct: 20, interest: 8.5, years: 20 }, emiGuideline: 0.35 },
  { key: "vacation", label: "Vacation", name: "Vacation", target: 150000, months: 10 },
  { key: "emergency", label: "Emergency fund", name: "Emergency fund", target: null, months: 12 },
  { key: "wedding", label: "Wedding", name: "Wedding", target: 1000000, months: 24 },
  { key: "education", label: "Education", name: "Higher education", target: 500000, months: 24 },
  { key: "gadget", label: "Laptop / phone", name: "New laptop", target: 80000, months: 6 },
  { key: "custom", label: "Custom goal", name: "My goal", target: 100000, months: 12 },
];

export const GOAL_TIPS = {
  car: [
    "Keep all car costs (EMI, fuel, insurance) under about 15% of your monthly income.",
    "A bigger down payment means a smaller loan and less interest paid overall.",
    "Compare on-road prices and end-of-quarter offers before you book.",
  ],
  home: [
    "Keep the home-loan EMI under about 35% of your take-home income.",
    "Budget 7–10% of the price extra for registration, stamp duty and interiors.",
    "Keep your emergency fund separate from the down payment.",
  ],
  emergency: [
    "Aim for 6 months of essential expenses.",
    "Keep it in a savings account, sweep FD or liquid fund so you can reach it quickly.",
  ],
  default: [
    "Set up an automatic transfer to a separate account on salary day so the money is saved first.",
    "A recurring deposit or liquid fund earns about 6–7% a year with low risk.",
  ],
};

// How much of each kind of spending can realistically be cut for a while.
const CUT_RULES = [
  [/shopping/i, 0.5, "Pause non-essential shopping for a while."],
  [/entertain|movie|leisure/i, 0.5, "Pick fewer paid outings and events."],
  [/travel|vacation/i, 0.5, "Postpone or shorten trips."],
  [/food|dining|restaurant|swiggy|zomato/i, 0.4, "Cook at home more; order food less often."],
  [/subscri|ott|stream/i, 0.5, "Cancel subscriptions you rarely use."],
  [/personal care|beauty|salon/i, 0.3, "Space out salon and personal-care spending."],
  [/^other|misc/i, 0.3, "Trim miscellaneous spending."],
  [/grocer/i, 0.12, "Plan meals and buy staples in bulk."],
  [/transport|fuel|cab|uber|ola/i, 0.2, "Use public transport or pool rides more often."],
  [/bill|utilit|electric|phone|internet|mobile/i, 0.1, "Move to cheaper plans and save electricity."],
  [/rent|emi|loan|insurance|health|medical|education|invest|tax|school/i, 0, null],
];

export function cutRule(category) {
  for (const [pattern, maxCut, tip] of CUT_RULES) {
    if (pattern.test(category)) return { maxCut, tip };
  }
  return { maxCut: 0.15, tip: "Trim where you can." };
}

const monthlyRate = (annualPct) => Math.max(Number(annualPct) || 0, 0) / 100 / 12;

export function requiredMonthlySaving(amount, months, annualPct = 0) {
  if (amount <= 0) return 0;
  const n = Math.max(1, Math.round(months));
  const r = monthlyRate(annualPct);
  return r === 0 ? amount / n : (amount * r) / ((1 + r) ** n - 1);
}

export function futureValue(monthly, months, annualPct = 0) {
  if (monthly <= 0 || months <= 0) return 0;
  const r = monthlyRate(annualPct);
  return r === 0 ? monthly * months : (monthly * ((1 + r) ** months - 1)) / r;
}

export function monthsToReach(amount, monthly, annualPct = 0) {
  if (amount <= 0) return 0;
  if (monthly <= 0) return Infinity;
  const r = monthlyRate(annualPct);
  return Math.ceil(r === 0 ? amount / monthly : Math.log(1 + (amount * r) / monthly) / Math.log(1 + r));
}

export function loanEmi(principal, annualPct, years) {
  if (principal <= 0) return 0;
  const n = Math.max(1, Math.round(years * 12));
  const r = monthlyRate(annualPct);
  return r === 0 ? principal / n : (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1);
}

const roundUp = (value, step = 100) => Math.ceil(value / step) * step;

// Spread a monthly saving gap over the categories that can be cut, wants before needs.
export function suggestCuts(expenses, gap) {
  const items = Object.entries(expenses)
    .map(([category, amount]) => ({ category, current: Number(amount) || 0, ...cutRule(category) }))
    .filter((item) => item.current > 0 && item.maxCut > 0)
    .map((item) => ({ ...item, capacity: item.current * item.maxCut, tier: item.maxCut >= 0.3 ? 1 : 2 }));
  const capacity = items.reduce((sum, item) => sum + item.capacity, 0);
  let remaining = Math.max(gap, 0);
  const cuts = [];
  for (const tier of [1, 2]) {
    if (remaining <= 0) break;
    const group = items.filter((item) => item.tier === tier);
    const groupCapacity = group.reduce((sum, item) => sum + item.capacity, 0);
    if (!groupCapacity) continue;
    const share = Math.min(remaining, groupCapacity);
    for (const item of group) {
      const saving = Math.min(item.current, roundUp((share * item.capacity) / groupCapacity, 50), roundUp(item.capacity, 50));
      if (saving > 0) cuts.push({ category: item.category, current: item.current, suggested: Math.max(item.current - saving, 0), saving, tip: item.tip });
    }
    remaining -= share;
  }
  cuts.sort((left, right) => right.saving - left.saving);
  return { cuts, total: cuts.reduce((sum, cut) => sum + cut.saving, 0), capacity };
}

export function planGoal({ target, saved = 0, months, annualReturn = 6, income, expenses, finance, preset }) {
  const price = Math.max(Number(target) || 0, 0);
  const financed = Boolean(finance?.enabled);
  const savingsTarget = financed ? price * (Number(finance.downPct) || 0) / 100 : price;
  const needed = Math.max(savingsTarget - (Number(saved) || 0), 0);
  const required = requiredMonthlySaving(needed, months, annualReturn);
  const totalExpenses = Object.values(expenses).reduce((sum, value) => sum + (Number(value) || 0), 0);
  const surplus = (Number(income) || 0) - totalExpenses;
  const gap = required - surplus;
  const { cuts, total: cutTotal, capacity } = gap > 0 ? suggestCuts(expenses, gap) : { cuts: [], total: 0, capacity: suggestCuts(expenses, Infinity).capacity };
  const remaining = Math.max(gap - cutTotal, 0);
  const plannedMonthly = Math.max(Math.min(required, Math.max(surplus, 0) + cutTotal), 0);
  const maxMonthly = Math.max(surplus, 0) + capacity;
  const goalDate = new Date();
  goalDate.setMonth(goalDate.getMonth() + Math.round(months));

  let loan = null;
  if (financed) {
    const principal = Math.max(price - savingsTarget, 0);
    const emi = loanEmi(principal, finance.interest, finance.years);
    const runningCost = preset === "car" ? price * 0.03 / 12 + 3000 : 0;
    const guideline = preset === "home" ? 0.35 : 0.15;
    const budgetAfter = (Number(income) || 0) - (totalExpenses - cutTotal) - emi - runningCost;
    // Highest price whose monthly cost (EMI + running cost) stays within the guideline.
    const costPerRupee = loanEmi(1, finance.interest, finance.years) * (1 - (Number(finance.downPct) || 0) / 100)
      + (preset === "car" ? 0.03 / 12 : 0);
    const fixedRunning = preset === "car" ? 3000 : 0;
    const affordablePrice = costPerRupee > 0 && income
      ? Math.max((guideline * income - fixedRunning) / costPerRupee, 0)
      : null;
    loan = {
      affordablePrice,
      principal,
      emi,
      runningCost,
      share: income ? (emi + runningCost) / income : null,
      guideline,
      totalInterest: emi * finance.years * 12 - principal,
      budgetAfter,
      affordable: income ? (emi + runningCost) / income <= guideline && budgetAfter >= 0 : false,
    };
  }

  return {
    price,
    savingsTarget,
    needed,
    required,
    surplus,
    gap,
    status: gap <= 0 ? "on_track" : remaining <= 0 ? "with_cuts" : "stretch",
    cuts,
    cutTotal,
    cutCapacity: capacity,
    remaining,
    plannedMonthly,
    goalDate,
    monthsAtCurrent: monthsToReach(needed, Math.max(surplus, 0), annualReturn),
    monthsWithMaxCuts: monthsToReach(needed, maxMonthly, annualReturn),
    affordableSavings: (Number(saved) || 0) + futureValue(maxMonthly, months, annualReturn),
    loan,
    projection: Array.from({ length: Math.round(months) + 1 }, (_, month) => ({
      month,
      saved: Math.round((Number(saved) || 0) + futureValue(plannedMonthly, month, annualReturn)),
      target: Math.round(savingsTarget),
    })),
  };
}
