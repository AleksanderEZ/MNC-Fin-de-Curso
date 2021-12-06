clear variables;
clc;

%File
attributes = 7;
instances = 210;
dataPath = "seeds_dataset.txt";
rowFormat = '%f %f %f %f %f %f %f %*d'; % discard class                 

seedsDataFile = fopen(dataPath, 'r');
seedsDataRaw = fscanf(seedsDataFile, rowFormat);
fclose("all");

%Data processing
X = reshape(seedsDataRaw, attributes, instances)'; %instances x attributes     

XC = centerValuesByColumn(X);

Z = (XC'*XC)/instances % Covariance matrix
[eigenVectors, eigenValues] = eig(Z);

[maxEigenValueIndex, secondmaxEigenValueIndex] = getTwoMaxIndices(eigenValues);

%Plotting
hold on

scatter(X(:, maxEigenValueIndex), X(:, secondmaxEigenValueIndex));

% Apply correct sizing to the vectors
vectors = eigenVectors;
for i=1:size(eigenVectors,1)
    vectors(i,:) = vectors(i,:) * eigenValues(i, i);
end

centerOfMass = [mean(X(:,maxEigenValueIndex)), mean(X(:,secondmaxEigenValueIndex))]
quiver([centerOfMass(1) ; centerOfMass(1)], [centerOfMass(2) ; centerOfMass(2)],  ...
    vectors([maxEigenValueIndex, secondmaxEigenValueIndex],maxEigenValueIndex), ...
    vectors([maxEigenValueIndex, secondmaxEigenValueIndex],secondmaxEigenValueIndex), 'LineWidth', 2)

ylabel("Coeficiente de asimetría");
xlabel("Longitud de la ranura del grano");

function centeredMatrix = centerValuesByColumn(X)
    centeredMatrix = X;
    attributes = size(centeredMatrix, 2);
    for col = 1:attributes
        currentColumn = centeredMatrix(:, col);
        columnMean = mean(centeredMatrix(:, col));
        centeredMatrix(:, col) =  currentColumn - columnMean;
    end
end

function [maxIndex, secondMaxIndex] = getTwoMaxIndices(eigenvaluesMatrix)
    eigenValues = max(eigenvaluesMatrix);
    [maxValue, maxIndex] = max(eigenValues);
    eigenValues(maxIndex) = [];
    [maxValue, secondMaxIndex] = max(eigenValues);
end