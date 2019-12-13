%
clear all; clc
data_matrix = [];
[c1, fs1] = audioread('Mozart - Requiem.mp3');
c1 = mean(c1, 2);
c1 = c1(find(c1, 1, 'first'):find(c1, 1, 'last'));
chunk = 5 * fs1;
num = floor(length(c1)/chunk);
c1 = reshape(c1(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, c1'];
[c2, fs2] = audioread('Mozart - The Marriage.mp3');
c2 = mean(c2, 2);
c2 = c2(find(c2, 1, 'first'):find(c2, 1, 'last'));
num = floor(length(c2)/chunk);
c2 = reshape(c2(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, c2'];
[c3, fs3] = audioread('Mozart - Rondo Alla Turca.mp3');
c3 = mean(c3, 2);
c3 = c3(find(c3, 1, 'first'):find(c3, 1, 'last'));
num = floor(length(c3)/chunk);
c3 = reshape(c3(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, c3'];

%%
[r1, fs4] = audioread('ACDC - Rock n Roll Train.mp3');
r1 = mean(r1, 2);
r1 = r1(find(r1, 1, 'first'):find(r1, 1, 'last'));
num = floor(length(r1)/chunk);
r1 = reshape(r1(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, r1'];
[r2, fs5] = audioread('ACDC - Rock or Bust.mp3');
r2 = mean(r2, 2);
r2 = r2(find(r2, 1, 'first'):find(r2, 1, 'last'));
num = floor(length(r2)/chunk);
r2 = reshape(r2(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, r2'];
[r3, fs6] = audioread('ACDC - TNT.mp3');
r3 = mean(r3, 2);
r3 = r3(find(r3, 1, 'first'):find(r3, 1, 'last'));
num = floor(length(r3)/chunk);
r3 = reshape(r3(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, r3'];

%%
[j1, fs7] = audioread('Dave Brubeck - Take Five.mp3');
j1 = mean(j1, 2);
j1 = j1(find(j1, 1, 'first'):find(j1, 1, 'last'));
num = floor(length(j1)/chunk);
j1 = reshape(j1(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, j1'];
[j2, fs8] = audioread('Giant Steps.mp3');
j2 = mean(j2, 2);
j2 = j2(find(j2, 1, 'first'):find(j2, 1, 'last'));
num = floor(length(j2)/chunk);
j2 = reshape(j2(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, j2'];
[j3, fs9] = audioread('Messengers - Moanin.mp3');
j3 = mean(j3, 2);
j3 = j3(find(j3, 1, 'first'):find(j3, 1, 'last'));
num = floor(length(j3)/chunk);
j3 = reshape(j3(1:num*chunk), [num, chunk]);
data_matrix = [data_matrix, j3'];

%%
labels = [];
str = ["Classical", "Rock", "Jazz"];
labels = [labels; repmat(str(1), 210, 1)];
labels = [labels; repmat(str(2), 129, 1)];
labels = [labels; repmat(str(3), 231, 1)];

%%
data_matrix = abs(fft(data_matrix));
[u, s, v] = svd(data_matrix, 'econ');

%%
a = diag(s).^2;
a = cumsum(a / sum(a));
plot(a)

%%
trainlabels = labels';
utrunc = u(:, 1:68);
traindata = utrunc'*data_matrix;
n = floor(size(traindata, 2)/(8));
c = randperm(size(traindata, 2), n);
testdata = traindata(:, c);
testlabels = trainlabels(:, c);
traindata(:,c) = [];
trainlabels(:,c) = [];

%%
model1 = fitcnb(traindata', trainlabels);
model2 = fitcecoc(traindata', trainlabels);
model3 = fitctree(traindata', trainlabels);