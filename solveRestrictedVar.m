function [x, partitionOut] = solveRestrictedVar(A, b, mu, partition, x0)
%SOLVERESTRICTEDVAR solve a restricted version of the group-based l2(l1) problem.
%   [x, partitionOut] = SOLVERESTRICTEDVAR(A, b, mu, partition, x0) finds the
%   minimum point of
%       0.5*||Ax-b||_2^2 + mu/2 * ||x||_l2(l1)^2
%   restricted to the support specified in partition. Input x0 (optional) is
%   for initialization.
%
%   Version exploiting the variational formulation of the norm-1.

XTOL = 1e-6; % minimum acceptable distance between two consecutive solution points
FUNTOL = 1e-6; % minimum acceptable distance between two consecutive solution values
DUALGAP = 1e-5; % minimum acceptable duality gap
BOUNDTOL = 1e-5; % acceptable tollerance when checking identities
MAX_ITER = 1000;
IGNOREDUALFEAS = true;

nFeatures = length(A(1,:));
if nargin == 4
    x0 = zeros(nFeatures, 1);
end

x = x0;
zInv = nan(nFeatures, 1);
modifyPartition = true(length(partition), 1);
iPartition = 0;
residual = b;
primal = 0;
for thisPartition = partition(modifyPartition)
    iPartition = iPartition + 1;
    indices = thisPartition.getindices('active');
    if isempty(indices)
        modifyPartition(iPartition) = false;
        continue;
    end

    primal = primal + sum(x(indices).^2 .* zInv(indices));
    residual = residual - A(:, indices) * x(indices);

    zInv(indices) = length(indices);
end
primal = 0.5 * norm(A*x - b)^2 + mu * primal / 2;

thisIter = 0;
hasConverged = false;
while ~hasConverged && thisIter < MAX_ITER
    thisIter = thisIter + 1;
    xold = x;
    primalOld = primal;

    primal = 0;
    for thisPartition = partition(modifyPartition)
        indices = thisPartition.getindices('active');

        for thisIndex = indices(:)'
            thisA = A(:, thisIndex);
            thisX = x(thisIndex);
            residual = residual + thisA * thisX;
            x(thisIndex) = thisA' * residual ...
                / (norm(thisA)^2 + mu * zInv(thisIndex));
            residual = residual - thisA * x(thisIndex);
        end

        % Variables z are used for the variational formulation of the norm.
        thisNorm = norm(x(indices), 1);
        z = abs(x(indices)) / thisNorm;
        tmp = 1./z;
        tmp(~isfinite(tmp)) = 0;
        zInv(indices) = tmp;

        primal = primal + sum(x(indices).^2 .* zInv(indices));
    end
    primal = 0.5 * norm(A*x - b)^2 + mu * primal / 2;

    dual = 0;
    u = A * x - b;
    for thisPartition = partition(modifyPartition)
        indices = thisPartition.getindices('active');

        lagrMult = mu * norm(x(indices), 1)^2 / 2;
        bound = sqrt(2 * mu * lagrMult);
        dual = dual + lagrMult + (1 / (norm(A(:, indices)' * u, inf) <= bound + BOUNDTOL) - 1);
    end
    dual = 0.5 * (norm(b)^2 - norm(u+b)^2) - dual;

    isDualOK = IGNOREDUALFEAS || isfinite(dual);
    hasConverged = isDualOK && (norm(xold - x) < XTOL * (1 + norm(xold)));
    hasConverged = hasConverged || isDualOK && (abs(primalOld - primal) < FUNTOL * (1 + abs(primalOld)));
    hasConverged = hasConverged || (abs(primal - dual) < DUALGAP);
end

% The solution is now computed: update the data stored in partition.
partitionNumbers = 1:length(partition);
partitionOut = partition;
for iPartition = partitionNumbers(modifyPartition)
    indices = partitionOut(iPartition).getindices;
    indicesOn = partitionOut(iPartition).getindices('active');
    partitionOut(iPartition) = Subset(x, indices);
    partitionOut(iPartition).activateIndices(indicesOn);
end
end
