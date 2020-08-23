syms x diam std

% define standard deviation function taken from JMP
std = @(x)2.4936441 + 0.4981291 * x(1) - 0.810586 * x(2)...
    +  0.7852652 * x(2) * x(2);

nonlcon = @constr % call the non-linear constraint defined in the function below

% minimise std based on the non-linear constraint
x = fmincon(std, [0.982 -0.65085714285], [], [], [], [],...
    [-1 -1],...
    [1 1],...
    nonlcon)

% convert x from coded parameter space to physical space
ap = x(1)*500 + 1000
oil = x(2)*1750 + 2250

% define constraint function
function [c,ceq] = constr(x)
    wanteddiam = 50
    c = []
    ceq = 39.963418 + 0.95277 * x(1) - 11.82268 * x(2)...
    - 3.002797 * x(1) * x(2)...
    + 2.7799737 * x(1) * x(1) - 1.875135 * x(2) * x(2)...
    - wanteddiam
end