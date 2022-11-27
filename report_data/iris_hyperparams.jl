using CSV, DataFrames
using Plots

file_loc = pwd() * "\\iris.csv";
data = DataFrame(CSV.File(file_loc, header = 1));
nrows, ncols = size(data);

tt_idx = Vector{Int64}(undef, 0);
lr_idx = Vector{Int64}(undef, 0);
ll_idx = Vector{Int64}(undef, 0);
for i = 1:nrows
    row = data[i, :]
    if row[:f1] == "t" && row[:f2] == "t"
        push!(tt_idx, i);
    elseif row[:f1] == "l" && row[:f2] == "l"
        push!(ll_idx, i);
    elseif row[:f1] == "l" && row[:f2] == "r"
        push!(lr_idx, i);
    end
end


records = [tt_idx, ll_idx, lr_idx];
funcs = [
    "f1: tanh, f2: tanh",
    "f1: logistic, f2: logistic",
    "f1: logistic, f2: relu"
];
filenames = ["tt", "ll", "lr"];

yticks = [0.0001, 0.0005, 0.001];
xticks = [0.01, 0.05, 0.1, 0.15]

for i in eachindex(records)
    func_records = data[records[i], :];
    p = scatter(
        func_records[!, :eta],
        func_records[!, :alpha],
        func_records[!, :test_acc],
        title = funcs[i],
        xlabel = "eta",
        ylabel = "alpha",
        zlabel = "Testing accuracy",
        legend = false,
        markercolor = :black,
        camera = (25, 10),
        xticks = xticks,
        yticks = yticks,
        gridalpha = 0.5
    );
    savefig(p, "./report/image/" * filenames[i] * ".png");
end
