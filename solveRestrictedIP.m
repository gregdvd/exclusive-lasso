function [x, partitionOut] = solveRestrictedIP(A, b, mu, partition, x0)
%SOLVERESTRICTEDIP solve a restricted version of the group-based l2(l1) problem.
%   [x, partitionOut] = SOLVERESTRICTEDIP(A, b, mu, partition, x0) finds the
%   minimum point of
%       0.5*||Ax-b||_2^2 + mu/2 * ||x||_l2(l1)^2
%   restricted to the support specified in partition. Input x0 (optional) is
%   for initialization.
%
%   Version exploiting the fmincon function of the Optimization Toolbox
%   (an implementation of the interior-point method).

nFeatures = length(A(1,:));
if nargin == 4
    x0 = zeros(nFeatures, 1);
end

lb = -inf(nFeatures, 1);
ub =  inf(nFeatures, 1);

nGroups = length(partition);
normMatrix = zeros(nGroups, nFeatures);
for iGroup = nGroups:-1:1
    groups{iGroup} = partition(iGroup).getindices;
    onIx = partition(iGroup).getindices('active');
    groupsOn{iGroup} = onIx;
    offIx = partition(iGroup).getindices('inactive');
    lb(offIx) = 0;
    ub(offIx) = 0;
    normMatrix(iGroup, onIx) = 1;
end

fun = @(w) 0.5*norm(b - A*w)^2 + mu/2 * sum((normMatrix * abs(w)).^2);
if ~verLessThan('matlab', '9.6')
    opts = optimoptions('fmincon', 'Display', 'notify-detailed', 'MaxFunctionEvaluations', 1e4, ...
        'StepTolerance', 1e-5);
else
    opts = optimoptions('fmincon', 'Display', 'notify-detailed', 'MaxFunEvals', 1e4, ...
        'TolX', 1e-5);
end
x = fmincon(fun, x0, [], [], [], [], lb, ub, [], opts);

% The solution is now computed: update the data stored in partition.
partitionOut(nGroups) = Subset;
for iGroup = 1:nGroups
    partitionOut(iGroup) = Subset(x, groups{iGroup});
    partitionOut(iGroup).activateIndices(groupsOn{iGroup});
end
end
