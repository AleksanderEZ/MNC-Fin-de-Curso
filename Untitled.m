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
centerOfMass = [mean(B(:,1)), mean(B(:,2))]
%plot(centerOfMass(1), centerOfMass(2), 'x', 'MarkerSize', 30, 'LineWidth', 8)
errorbar(1, centerOfMass(2), Z(2,2))
errorbar(centerOfMass(1), 1, Z(1,1), 'horizontal')
quiver([centerOfMass(1) centerOfMass(1)], [centerOfMass(2) centerOfMass(2)], eigenVectors(1,:), eigenVectors(2,:),0, 'LineWidth', 2);
xlabel("X variance")
ylabel("Y variance")
axis([0 20 0 20])

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