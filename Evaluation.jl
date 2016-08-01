function entropy(partitionSet, N)
    return mapreduce(x -> -(length(x) / N) * log(length(x) / N), +, partitionSet)
end

function getPartitionSet(vector, nrOfClusters)
    d = Dict(zip(1:nrOfClusters,[Int[] for _ in 1:nrOfClusters]));
    for i in 1:length(vector)
        push!(d[vector[i]], i)
    end
    return collect(values(d))
end


function mutualInformation(partitionSetA, partitionSetB, N)


    anonFunc = (x) -> (
                intersection = size(intersect(partitionSetA[x[1]], partitionSetB[x[2]]),1);
                return intersection > 0 ?
                        (intersection/N) * log((intersection * N) / (size(partitionSetA[x[1]],1) * size(partitionSetB[x[2]],1))) : 0;
    )

    mapreduce(anonFunc, + , product(1:length(partitionSetA),1:length(partitionSetB)))
end


function normalizedMutualInformation(partitionSetA, partitionSetB, N)
    mutualInformation(partitionSetA, partitionSetB, N) / ((entropy(partitionSetA, N)  + entropy(partitionSetB, N)) / 2)
end
