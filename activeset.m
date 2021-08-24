function x = activeset(A, b, mu, groups, x0)
%ACTIVESET active set algorithm for the group-based l2(l1) norm.
%   x = ACTIVESET(A, b, mu, groups) finds the minimum point of
%       0.5*||Ax-b||_2^2 + mu/2 * ||x||_l2(l1)^2
%   by means of the active-set algorithm. Here, ||x||_l2(l1) is the l2 norm of a
%   vector of positive reals obtained by computing the l1 norm of all groups.
%   Input groups is a cell of arrays, each array containing the indices of a
%   different group.
%
%   x = ACTIVESET(A, b, mu, groups, x0) initializes the algorithm with the
%   solution guess x0.

DUALITYGAP = 0.05; % max duality gap allowed
THRESHOLD = 1e-5; % minimum allowable value of active entries
solveRestricted = @solveRestrictedVar;
% Use suffix IP for interior-point, Var for variational formulation.

[nObservations, nFeatures] = size(A);
if nargin == 4
    x0 = zeros(nFeatures, 1);
end
nGroups = length(groups);
usedEntries = false(nFeatures, 1);

counts = zeros(nFeatures, 1);
partition(nGroups) = Subset;

for iGroup = 1:nGroups
    thisGroup = groups{iGroup};
    counts(thisGroup) = counts(thisGroup) + 1;
    partition(iGroup) = Subset(x0, thisGroup);
    ixActive = thisGroup(abs(x0(thisGroup)) > 0);
    usedEntries(ixActive) = true;
    partition(iGroup).activateIndices(ixActive);
end
assert(all(counts == 1), 'groups should describe a partition of the indices of x');

currentGradient = A' * (A * x0 - b);
[passed, nextActive] = checknecessary(currentGradient, partition, mu);
x = x0;
while ~passed && sum(usedEntries) < nObservations/2
    usedEntries(nextActive.index) = true;
    ixtmp = partition(nextActive.group).getindices('active');
    partition(nextActive.group).activateIndices([ixtmp; nextActive.index]);
    [x, partition] = solveRestricted(A, b, mu, partition, x);
    currentGradient = A' * (A * x - b);
    [passed, nextActive] = checknecessary(currentGradient, partition, mu);
end

[passed, nextActive] = checksufficient(currentGradient, partition, mu, DUALITYGAP);
while ~passed && sum(usedEntries) < nObservations/2
    usedEntries(nextActive.index) = true;
    ixtmp = partition(nextActive.group).getindices('active');
    partition(nextActive.group).activateIndices([ixtmp; nextActive.index]);
    [x, partition] = solveRestricted(A, b, mu, partition, x);
    currentGradient = A' * (A * x - b);
    [passed, nextActive] = checksufficient(currentGradient, partition, mu, DUALITYGAP);
end
% x = x .* (abs(x) > THRESHOLD);

function [passed, nextActive] = checknecessary(currentGradient, partition, mu)
% Check whether the necessary condition for the solution is met and suggest the
% next entry to activate.

passed = true;
nGroups = length(partition);
maxMargin = -inf;
nextActive = [];

for iGroup = 1:nGroups
    thisPartition = partition(iGroup);
    inactiveIndices = thisPartition.getindices('inactive');
    inactiveGradient = currentGradient(inactiveIndices);
    localNormOne = thisPartition.getnormone;

    if localNormOne > 0
        margins = abs(inactiveGradient) / localNormOne - mu;
    else
        margins = abs(inactiveGradient);
    end
    passed = passed && all(margins <= 0);

    [tmpMax, tmpIx] = max(margins);
    if tmpMax > maxMargin
        maxMargin = tmpMax;
        nextActive.index = inactiveIndices(tmpIx(1));
        nextActive.group = iGroup;
    end
end


function [passed, nextActive] = checksufficient(currentGradient, partition, mu, gap)
% Check whether the sufficient condition for the solution is met and suggest the
% next entry to activate.
nGroups = length(partition);
maxMargin = -inf;
nextActive = [];
totalGapOff = 0;
totalGapInact = 0;

normSqrd = 0;

for iGroup = 1:nGroups
    thisPartition = partition(iGroup);
    inactiveIndices = thisPartition.getindices('inactive');
    inactiveGradient = currentGradient(inactiveIndices);

    normSqrd = normSqrd + thisPartition.getnormone^2;

    if isempty(thisPartition.getindices('active'))
        % Conjecture: this should never fail, if it does take it out, not proven.
        assert(all(inactiveGradient == 0));
        continue;
    end
    if isempty(inactiveIndices)
        continue;
    end

    [tmpMax, tmpIx] = max(abs(inactiveGradient));
    if tmpMax > maxMargin
        maxMargin = tmpMax;
        nextActive.index = inactiveIndices(tmpIx(1));
        nextActive.group = iGroup;
    end

    if ~isempty(thisPartition.getindices('active'))
        totalGapOff = totalGapOff + tmpMax^2;
    else
        totalGapInact = totalGapInact + tmpMax^2;
    end
end
passed = (totalGapOff - mu^2 * normSqrd <= 2 * mu * gap) ...
    && (totalGapInact - mu^2 * normSqrd <= 2 * mu * gap);
