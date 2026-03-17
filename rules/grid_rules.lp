% ==========================================
% Austrian Grid Safety Rules
% ==========================================

% 1. Non-Negative Generation Constraint
% Total generation cannot physically be less than zero.
anomaly(T, "Negative Generation Predicted") :- pred(T, V), V < 0.

% 2. Baseline Load Constraint
% The Austrian grid total generation rarely drops below a certain baseline at night.
% Let's flag any prediction below 20 MW as highly suspicious.
anomaly(T, "Predicted Below Historical Grid Baseline") :- pred(T, V), V < 20.

% 3. Physical Ramp-Rate Limit (normal workdays only)
% The total grid generation cannot jump by extreme amounts in just 15 minutes 
% due to the mechanical inertia of turbines. (Assume a 25 MW max jump for this subset).
% 'not holiday(T)' uses negation-as-failure so this rule is skipped on public holidays.
anomaly(T, "Ramp Rate Violation (Spike)") :- pred(T, V1), pred(T-1, V0), V1 - V0 > 25, not holiday(T).
anomaly(T, "Ramp Rate Violation (Drop)") :- pred(T, V1), pred(T-1, V0), V0 - V1 > 25, not holiday(T).

% 4. Prediction Error Veto (normal workdays only)
% If the AI prediction deviates from the actual value by more than 30 MW, flag it as an AI failure.
% On public holidays, the model is less accurate by design — handled by rule 5 instead.
anomaly(T, "Critical AI Deviation") :- pred(T, P), actual(T, A), P - A > 30, not holiday(T).
anomaly(T, "Critical AI Deviation") :- pred(T, P), actual(T, A), A - P > 30, not holiday(T).

% 5. Holiday-Aware Relaxed Deviation Threshold
% On Austrian public holidays, demand patterns differ significantly from workdays.
% Apply a more lenient 45 MW threshold — only holiday(T) timesteps reach here
% (rules 3/4 are suppressed for holidays via 'not holiday(T)' above).
anomaly(T, "Critical AI Deviation") :- pred(T, P), actual(T, A), P - A > 45, holiday(T).
anomaly(T, "Critical AI Deviation") :- pred(T, P), actual(T, A), A - P > 45, holiday(T).

% 6. Holiday Ramp-Rate Relaxation
% Grid operators pre-schedule different ramp profiles on public holidays.
% Raise the ramp-rate anomaly threshold to 40 MW on holiday timesteps.
anomaly(T, "Ramp Rate Violation (Spike)") :- pred(T, V1), pred(T-1, V0), V1 - V0 > 40, holiday(T).
anomaly(T, "Ramp Rate Violation (Drop)") :- pred(T, V1), pred(T-1, V0), V0 - V1 > 40, holiday(T).

% Display only the anomalies in the final output
#show anomaly/2.