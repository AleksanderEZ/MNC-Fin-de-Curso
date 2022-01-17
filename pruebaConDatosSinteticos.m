clear variables;
clc;

m = 500; n = 2;
A = randn(m,n);
% deformación por un factor de 3
A(:,2) = 3*A(:,2);
% matriz de rotación
phi = 45;
cose = cosd(phi); sen = sind(phi);
R = [cose -sen; sen cose];
% rotación y traslacion al punto (10,10)
B = A*R + 10;

XC = centerValuesByColumn(B);

Z = (XC'*XC)/m; %Covariance matrix
[eigenVectors, eigenValues] = eig(Z);

scatter(B(:, 1), B(:, 2));
hold on;
centerOfMass = [mean(B(:,1)), mean(B(:,2))];
vectors = eigenVectors;
vectors(1,:) = vectors(1,:) * eigenValues(1, 1);
vectors(2,:) = vectors(2,:) * eigenValues(2, 2);
quiver([centerOfMass(1) ; centerOfMass(1)], [centerOfMass(2) ; ...
    centerOfMass(2)], vectors(:,1), vectors(:,2), 'LineWidth', 2)
errorbar(1, centerOfMass(2), Z(2,2))
errorbar(centerOfMass(1), 1, Z(1,1), 'horizontal')
xlabel("X variance")
ylabel("Y variance")

b = B*eigenVectors
figure(2);
scatter(b(:, 1), b(:, 2));


function centeredMatrix = centerValuesByColumn(X)
    centeredMatrix = X;
    attributes = size(centeredMatrix, 2);
    for col = 1:attributes
        currentColumn = centeredMatrix(:, col);
        columnMean = mean(centeredMatrix(:, col));
        centeredMatrix(:, col) =  currentColumn - columnMean;
    end
end