% reference_values.m
% -------------------------------------------------------------------------
% Generate MATLAB gapmetric (Robust Control Toolbox) reference values for the
% nugap package cross-check. The Vinnicombe nu-gap is the SECOND output of
% gapmetric:  [gap, nugap] = gapmetric(P1, P2).
%
% Requires: Control System Toolbox + Robust Control Toolbox (for gapmetric),
%           MATLAB R2016b+ (for jsondecode).
%
% Usage, from this folder:
%       >> reference_values
% Produces reference_values.csv (columns: id, gap, nugap). Then run the Python
% harness:  python crosscheck_matlab.py
% -------------------------------------------------------------------------

panel = jsondecode(fileread('panel.json'));
n = numel(panel);

ids    = strings(n,1);
gaps   = zeros(n,1);
nugaps = zeros(n,1);

rowvec = @(v) reshape(double(v), 1, []);   % JSON arrays decode to columns; force row

fprintf('%-26s %10s %10s\n', 'id', 'gap', 'nugap');
for k = 1:n
    c  = panel(k);
    n1 = rowvec(c.num1);  d1 = rowvec(c.den1);
    n2 = rowvec(c.num2);  d2 = rowvec(c.den2);

    isContinuous = isempty(c.dt) || (isnumeric(c.dt) && all(isnan(c.dt)));
    if isContinuous
        P1 = tf(n1, d1);          P2 = tf(n2, d2);
    else
        Ts = double(c.dt);
        P1 = tf(n1, d1, Ts);      P2 = tf(n2, d2, Ts);
    end

    [g, ng] = gapmetric(P1, P2);   % nugap = Vinnicombe nu-gap (2nd output)

    ids(k)    = string(c.id);
    gaps(k)   = g;
    nugaps(k) = ng;
    fprintf('%-26s %10.6f %10.6f\n', c.id, g, ng);
end

T = table(ids, gaps, nugaps, 'VariableNames', {'id', 'gap', 'nugap'});
writetable(T, 'reference_values.csv');
fprintf('\nWrote reference_values.csv (%d rows).\n', n);
