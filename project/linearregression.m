clear
load affairs
x1 = table2array(Affairs(:,4));
x2 = table2array(Affairs(:,2));
y = table2array(Affairs(:,5));

X = [ones(size(x1)) x1 x2 x1.*x2];
b = regress(y,X)    % Removes NaN data

scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):1:max(x1);
x2fit = min(x2):1:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Age')
ylabel('Affairs')
zlabel('Years Married')
view(50,10)
hold off