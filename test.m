% System parameters
nObservations = 140;
nFeatures = 200;
SNR = 15; % dB
nLambdas = 50;

% Signal support consists of 2 strings of 10 active entries each: from position
% 4 to position 13 and from position 73 to position 82.
stringLength = 10; % should be a divisor of nFeatures
activeBlocks = [4, 73];
activeEntries = false(nFeatures, 1);
for thisBlock = activeBlocks
    activeEntries(thisBlock + (0:stringLength-1)) ...
        = activeEntries(thisBlock + (0:stringLength-1)) | true(stringLength, 1);
end
x = zeros(nFeatures, 1);
x(activeEntries) = 1;

% Create measurement matrix
A = randn(nObservations, nFeatures);
A = A * diag(1./sqrt(sum(A.^2)));

% Groups for the exclusive lasso: the number of groups is equal to the length of
% the strings. Group i consists of all entries with index congruent to i modulo
% nGroups == stringLength (in this way, each string activates one entry per
% group).
nGroups = stringLength;
groups{nGroups} = [];
for iGroup = 1:nGroups
    groups{iGroup} = iGroup:nGroups:nFeatures;
end

% Groups for the group lasso with overlaps: there are nFeatures groups of length
% stringLength, starting at each entry.
tc = [1; zeros(nFeatures - stringLength, 1); ones(stringLength - 1, 1)];
tr = [ones(1, stringLength) zeros(1, nFeatures - stringLength)];
groupIndicator = toeplitz(tc, tr) > 0.2;

% Generate noise
noise = randn(nObservations, 1) / sqrt(nObservations) * 10^(-SNR/20);
observations = A * x + noise;

%%
% Space the values of lambda (also used for mu) logarithmically.
lambdaMax = max(abs(A' * observations)) * 2;
logLambdaMin = -2.3;
logLambdaMax = log10(lambdaMax);
lambdas = logspace(logLambdaMin, logLambdaMax, nLambdas);

% Matrices to save results
xProx = nan(nFeatures, nLambdas); % exclusive lasso: proximal
xAS = nan(nFeatures, nLambdas); % exclusive lasso: active set
xAT = nan(nFeatures, nLambdas); % exclusive lasso: active string
xLasso = nan(nFeatures, nLambdas); % classic lasso (norm 1)
xOverlap = nan(nFeatures, nLambdas); % group lasso with overlaps

% tmp variables for current state
xPrev = zeros(nFeatures, 1);
xPrevL = zeros(nFeatures, 1);
xPrevO = zeros(nFeatures, 1);

for iLambda = nLambdas:-1:1
    fprintf('Point %3d of %3d.\n', nLambdas - iLambda + 1, nLambdas);
    xProx(:, iLambda) = fista(A, observations, lambdas(iLambda), groups, xPrev);
    xPrev = xProx(:, iLambda);
    xAS(:, iLambda) = activeset(A, observations, lambdas(iLambda), groups);
    xAT(:, iLambda) = activestrings(A, observations, lambdas(iLambda), groups, stringLength);
    xLasso(:, iLambda) = fistabasic(A, observations, lambdas(iLambda), xPrevL);
    xPrevL = xLasso(:, iLambda);
    xOverlap(:, iLambda) = fistaOverlap(A, observations, lambdas(iLambda), groupIndicator, xPrevO);
    xPrevO = xOverlap(:, iLambda);
end

%%
figure
imagesc([logLambdaMin logLambdaMax], [1 200], abs(xProx))
title('Proximal')
ylabel('Parameter Index')
xlabel('log \lambda')

figure
imagesc([logLambdaMin logLambdaMax], [1 200], abs(xAS))
title('Active Set')
ylabel('Parameter Index')
xlabel('log \mu')

figure
imagesc([logLambdaMin logLambdaMax], [1 200], abs(xAT))
title('Active Strings')
ylabel('Parameter Index')
xlabel('log \mu')

figure
imagesc([logLambdaMin logLambdaMax], [1 200], abs(xLasso))
title('Lasso (norm-1)')
ylabel('Parameter Index')
xlabel('log \lambda')

figure
imagesc([logLambdaMin logLambdaMax], [1 200], abs(xOverlap))
title('Group Lasso with Overlaps')
ylabel('Parameter Index')
xlabel('log \lambda')