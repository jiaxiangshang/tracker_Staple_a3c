u = 2;
theta = 0.5;
numbers = 1000;
datas = lognrnd(u, theta, [1, numbers]);

[f,xi] = ksdensity(datas);
plot(xi,f,'--r','LineWidth',1.5) 