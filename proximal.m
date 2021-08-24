function p = proximal(x, groups)
%PROXIMAL operator for the group-based l2(l1) norm.
%   p = PROXIMAL(x, groups) computes the proximal point p to vector x according
%   to the l2(l1) norm defined on groups. x is a vector of reals, groups is a
%   cell of arrays, each array containing the indices of a different group.

nTotalEntries = length(x);
nGroups = length(groups);
usedEntries = false(nTotalEntries, 1);
entryGroup = nan(nTotalEntries, 1);

counts = zeros(nTotalEntries, 1);
partition(nGroups) = Subset;
partitionNormOnes = nan(nGroups, 1);
partitionNActives = nan(nGroups, 1);

for iGroup = 1:nGroups
    entryGroup(groups{iGroup}) = iGroup;
    counts(groups{iGroup}) = counts(groups{iGroup}) + 1;
    partition(iGroup) = Subset(x, groups{iGroup});
    ix = partition(iGroup).activateNext;
    usedEntries(ix) = true;
    partitionNormOnes(iGroup) = partition(iGroup).getnormone;
    partitionNActives(iGroup) = partition(iGroup).getnactive;
end
assert(all(counts < 2) && all(counts > 0),...
    'groups should describe a partition of the indices of x');

% Compute the multiplier for the case only one element per group is active.
sumSqNorms = sum((partitionNormOnes ./ partitionNActives).^2);
mu = -1 + sqrt(sumSqNorms);

% Check the solution and look for the group that would cause the next "change of piece."
[isSolved, mask] = checksolution(partition, mu);

while ~isSolved
    % Activate the next highest absolute entry in the group identified above
    % (it may happen that more than one group exceed their upperbound by the
    % same quantity.
    grouplist = 1:nGroups;
    tmpGroups = grouplist(mask);
    for thisGroup = tmpGroups
        ix = partition(thisGroup).activateNext;
        usedEntries(ix) = true;
        partitionNormOnes(thisGroup) = partition(thisGroup).getnormone;
        partitionNActives(thisGroup) = partition(thisGroup).getnactive;
    end

    mu = computeMu(partitionNormOnes, partitionNActives, mu);

    % Check the solution and look for the group that would cause the next "change of piece."
    [isSolved, mask] = checksolution(partition, mu);
end

if mu > 0
    thresholds = partitionNormOnes ./ (partitionNActives + mu);
    p = nan(nTotalEntries, 1);
    for iGroup = 1:nGroups
        % soft thresholding
        mask = (entryGroup == iGroup);
        tmp = x(mask);
        p(mask) = (tmp - thresholds(iGroup) * sign(tmp)) .* (abs(tmp) > thresholds(iGroup));
    end
else % if mu < 0 there is no thresholding
    p = x;
end
end


function mu = computeMu(partitionNormOnes, partitionNActives, muOld)
% Compute the value of the Lagrange multiplier by means of the Newton-Raphson method.
    tolerance = 1e-4;
    mu = muOld;
    fOld = 1 - sum(partitionNormOnes.^2 ./ (mu + partitionNActives).^2);
    fPrimeOld = 2 * sum(partitionNormOnes.^2 ./ (mu + partitionNActives).^3);

    while abs(fOld) > tolerance
        mu = muOld - fOld / fPrimeOld;
        fOld = 1 - sum(partitionNormOnes.^2 ./ (mu + partitionNActives).^2);
        fPrimeOld = 2 * sum(partitionNormOnes.^2 ./ (mu + partitionNActives).^3);
        muOld = mu;
    end
end


function [isSolved, mask] = checksolution(partition, mu)
% Check whether a solution has been reached. If not, mark the group(s) with the
% next activation.
    nGroups = length(partition);

    isSolved = true;
    mask = false(nGroups, 1);
    minNextPiece = inf;
    for iGroup = 1:nGroups
        [isGroupOk, ~, ub] = partition(iGroup).checksolution(mu);
        isSolved = isSolved && isGroupOk;

        nextPiece = ub + mu;
        if nextPiece < minNextPiece
            minNextPiece = nextPiece;
            mask = false(nGroups, 1);
            mask(iGroup) = true;
        elseif nextPiece == minNextPiece
            mask(iGroup) = true;
        end
    end
end
