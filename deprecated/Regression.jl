"""
    X = rand(100, 2)
    y = X[:,1] +X[:, 2]

    m = RegModel()
    fit!(m, X, y)
    predict(m, X)
"""
module Regression

using ScikitLearn
@sk_import neighbors: KNeighborsRegressor
@sk_import model_selection: GridSearchCV

type RegModel
  models::Vector
end

function RegModel()
  knn = GridSearchCV(KNeighborsRegressor(), cv=5,
  param_grid = Dict("n_neighbors"=>[4, 5, 10, 20, 50]))
  RegModel([knn])
end

import StatsBase.fit!
function fit!(rm::RegModel, X, y)
  for m in rm.models
    ScikitLearn.fit!(m, X, y)
  end
end

import StatsBase.predict
function predict(rm::RegModel, X)
    y_pred = mean([ScikitLearn.predict(m, X) for m in rm.models])
end

end
