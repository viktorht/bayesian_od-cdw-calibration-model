from predict_cdw.mean_model_predict import *

def test_posterior_predictive_dist():
    test_od = np.array([1, 60, 50, 100])
    result = posterior_predictive_dist(test_od)

    assert result.shape[1] == 4, 'predictions results have the wrong shape. Wrong number of columns.'

test_posterior_predictive_dist()