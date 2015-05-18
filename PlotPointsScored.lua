require 'gnuplot'

gamescores = torch.load("TimePunishedQlearning.t7")
scatterScores = {}
for i = 1, table.getn(gamescores) do
	scatterScores[i] = {i, gamescores[i]}
end
scatterScores = torch.Tensor(scatterScores)
print(scatterScores)
gnuplot.plot(scatterScores, '+')
print(torch.mean(torch.Tensor(gamescores)))