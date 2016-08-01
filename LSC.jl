using Clustering;
using Distances;

function getLandmarks(X, p, method=:Random)
    if(method == :Random)
        numberOfPoints = size(X,2);
        landmarks = X[:,randperm(numberOfPoints)[1:p]];
        return landmarks;
    end

    if(method == :Kmeans)
        kmeansResult = kmeans(X,p)
        return kmeansResult.centers;
    end

    throw(ArgumentError("method can only be :Kmeans or :Random"));
end

function gaussianKernel(distance, bandwidth)
    exp(-distance / (2*bandwidth^2));
end


function getLinearCoding(X, landmarks, bandwidth, r)
    distances = pairwise(Distances.Euclidean(), landmarks, X);
    similarities = map(x -> gaussianKernel(x, bandwidth), distances);
    ZHat = spzeros(size(similarities,1), size(similarities,2));

    for i in 1:(size(similarities,2))
        topLandMarksIndices = selectperm(similarities[:,i], 1:r, rev=true);
        topLandMarksCoefficients = similarities[topLandMarksIndices, i];
        topLandMarksCoefficients = topLandMarksCoefficients / sum(topLandMarksCoefficients);
        ZHat[topLandMarksIndices,i] = topLandMarksCoefficients;
    end
    return diagm(sum(ZHat,2)[:])^(-1/2) * ZHat;
end


function LSCClustering(X, nrOfClusters, nrOfLandmarks, method, nonZeroLandmarkWeights, bandwidth)
    landmarks = getLandmarks(X, nrOfLandmarks, method)
    ZHat = getLinearCoding(X, landmarks, bandwidth, nonZeroLandmarkWeights)
    svdResult = svd(transpose(ZHat))
    clusteringResult = kmeans(transpose(svdResult[1][:,1:nrOfClusters]),nrOfClusters);
    return clusteringResult
end
