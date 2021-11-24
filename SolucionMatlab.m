clear variables;
clc;

attributes = 7;
instances = 210;
dataPath = "seeds_dataset.txt";
rowFormat = '%f %f %f %f %f %f %f %*d'; %discard class                 

seedsDataFile = fopen(dataPath, 'r');
seedsDataRaw = fscanf(seedsDataFile, rowFormat);
fclose("all");

X = reshape(seedsDataRaw, attributes, instances)'; %instances x attributes     

XC = centerValuesByColumn(X);

Z = (XC'*XC)/instances; %Covariance matrix
[eigenVectors, eigenValues] = eig(Z);



function centeredMatrix = centerValuesByColumn(X)
    centeredMatrix = X;
    attributes = size(centeredMatrix);
    attributes = attributes(2);
    for col = 1:attributes
        currentColumn = centeredMatrix(:, col);
        columnMean = mean(centeredMatrix(:, col));
        centeredMatrix(:, col) =  currentColumn - columnMean;
    end
end