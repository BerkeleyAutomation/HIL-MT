import collections

import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.pipeline as pl
import sklearn.preprocessing as pp

from utils import DictTree

DEBUG = False

NUM_FOLDS = 20
MIN_SUB_COUNT = 1
C = 100.


def catalog(config):
    """

    Args:
        config (DictTree)
    """
    if '|' in config.name:
        return ModelSelector(config)
    elif config.name.startswith('t_'):
        return TimeDependentModel(DictTree(
            name=config.name[len('t_'):],
            cnt_idx=config.arg_in_len,
            max_cnt=config.max_cnt,
            num_sub=config.num_sub,
        ))
    elif config.name.startswith('log_lin'):
        return LogisticLinearModel(DictTree(
            num_sub=config.num_sub,
        ))
    elif config.name.startswith('log_poly'):
        return LogisticPolynomialModel(DictTree(
            num_sub=config.num_sub,
            degree=int(config.name[len('log_poly'):]),
        ))
    elif config.name.startswith('log_mlp'):
        return LogisticMLPModel(DictTree(
            num_sub=config.num_sub,
            degree=int(config.name[len('log_mlp'):config.name.index('[')]),
            hidden_sizes=[int(s) for s in config.name[config.name.index('[') + 1:-1].split(', ')],
        ))
    elif config.name == 'table':
        return TableModel(config)
    else:
        raise NotImplementedError(config.name)


def validate(model, data, sub_arg_accuracy=None):
    """

    Args:
        model
        data (DictTree)
        sub_arg_accuracy
    """
    pred = model.predict(data.iput)
    sub_corr = (pred.sub == data.oput.sub)  # type: np.ndarray
    sub_corr = sub_corr.all()
    arg_mse = ((pred.arg - data.oput.arg) ** 2).sum(0)
    if DEBUG:
        arg_rmse = (arg_mse / len(data.iput)) ** .5
        # arg_corr = (arg_rmse <= sub_arg_accuracy).all()
        print(data.iput)
        print(data.oput.sub)
        print(pred.sub)
        print(data.oput.arg)
        print(pred.arg)
        print(sub_corr)
        print(arg_rmse)
    if sub_arg_accuracy is None:
        return DictTree(data_len=len(data.iput), sub_corr=sub_corr, arg_mse=arg_mse)
    else:
        arg_corr = ((arg_mse / len(data.iput)) ** .5 <= sub_arg_accuracy).all()
        return sub_corr and arg_corr


def total_validation(validation, sub_arg_accuracy):
    data_len = sum(v.data_len for v in validation)
    sub_corr = all(v.sub_corr for v in validation)
    arg_mse = sum(v.arg_mse for v in validation)
    arg_rmse = (arg_mse / data_len) ** .5
    arg_corr = (arg_rmse <= sub_arg_accuracy).all()
    if DEBUG:
        print("Validation: sub = {}; arg = {} ({})".format(sub_corr, arg_rmse, arg_corr))
    return sub_corr and arg_corr


class ModelSelector(object):
    def __init__(self, config):
        """

        Args:
            config (DictTree)
        """
        self.sub_arg_accuracy = config.sub_arg_accuracy
        self.models = [catalog(config | DictTree(name=name)) for name in config.name.split('|')]
        self.selector = None

    def __repr__(self):
        return 'ModelSelector({})'.format(self.models[self.selector or 0])

    def fit(self, data):
        """

        Args:
            data (DictTree)
        """
        for selector, model in enumerate(self.models):
            num_folds = min(len(data.iput), NUM_FOLDS)
            kf = ms.KFold(num_folds, True)
            for train_idxs, valid_idxs in kf.split(data.iput):
                train_data = self._split_data(data, train_idxs)
                valid_data = self._split_data(data, valid_idxs)
                model.fit(train_data)
                validated = validate(model, valid_data, self.sub_arg_accuracy)
                if not validated:
                    break
            else:
                model.fit(data)
                self.selector = selector
                break
        else:
            self.models[0].fit(data)
            self.selector = None

    def predict(self, iput):
        return self.models[self.selector or 0].predict(iput)

    @staticmethod
    def _split_data(data, idxs):
        """

        Args:
              data (DictTree)
        """
        return DictTree(
            iput=data.iput[idxs],
            oput=DictTree(
                sub=data.oput.sub[idxs],
                arg=data.oput.arg[idxs],
            ),
        )


class TimeDependentModel(object):
    def __init__(self, config):
        """

        Args:
            config (DictTree)
        """
        self.models = [catalog(config) for _ in range(config.max_cnt)]
        self.cnt_idx = config.cnt_idx
        self.max_cnt = config.max_cnt

    def fit(self, data):
        """

        Args:
            data (DictTree)
        """
        step_idxs = [[] for _ in range(self.max_cnt)]
        for step_idx, iput in enumerate(data.iput):
            step_idxs[int(iput[self.cnt_idx])].append(step_idx)
        for cnt in range(self.max_cnt):
            if step_idxs[cnt]:
                iput = data.iput[step_idxs[cnt]]
                sub = data.oput.sub[step_idxs[cnt]]
                arg = data.oput.arg[step_idxs[cnt]]
            else:
                iput = np.zeros((1, len(data.iput[0])))
                sub = np.zeros(1, np.int32)
                arg = np.zeros((1, len(data.oput.arg[0])))
            self.models[cnt].fit(DictTree(
                iput=iput,
                oput=DictTree(
                    sub=sub,
                    arg=arg,
                ),
            ))

    def predict(self, iput):
        oput = DictTree(
            sub=[],
            arg=[],
        )
        for i in iput:
            pred = self.models[int(i[self.cnt_idx])].predict([i])
            oput.sub.extend(pred.sub)
            oput.arg.extend(pred.arg)
        return oput


class TwoStepModel(object):
    def __init__(self, config):
        """

        Args:
            config (DictTree)
        """
        self.num_sub = config.num_sub
        self.const_sub = None
        self.sub_model = self._make_sub_model(config)
        self.arg_models = [self._make_arg_model(config) for _ in range(self.num_sub)]
        self.fitted = False

    def _make_sub_model(self, config):
        raise NotImplementedError

    def _make_arg_model(self, config):
        raise NotImplementedError

    def fit(self, data):
        sub_counts = np.bincount(data.oput.sub)
        if sum(sub_counts >= MIN_SUB_COUNT) < 2:
            self.const_sub = sub_counts.argmax()
        else:
            self.const_sub = None
            self.sub_model.fit(data.iput, data.oput.sub)
            self.fitted = True
        pred_sub = self.predict(data.iput, only_sub=True)  # type: collections.Sequence[int]
        step_idxs = [[] for _ in range(self.num_sub)]
        for step_idx, sub in enumerate(pred_sub):
            step_idxs[sub].append(step_idx)
        for sub in range(self.num_sub):
            if step_idxs[sub]:
                self.arg_models[sub].fit(data.iput[step_idxs[sub]], data.oput.arg[step_idxs[sub]])
            else:
                self.arg_models[sub].fit(np.zeros((1, len(data.iput[0]))), np.zeros((1, len(data.oput.arg[0]))))

    def predict(self, iput, only_sub=False):
        if self.const_sub is None:
            pred_sub = self.sub_model.predict(iput)
        else:
            pred_sub = np.full(len(iput), self.const_sub)
        if only_sub:
            return pred_sub
        else:
            pred_arg = []
            for inp, sub in zip(iput, pred_sub):
                arg = self.arg_models[sub].predict([inp])
                pred_arg.extend(arg)
            return DictTree(sub=pred_sub, arg=pred_arg)


class LogisticLinearModel(TwoStepModel):
    def _make_sub_model(self, config):
        if config.get('cv_reg'):
            logistic = lm.LogisticRegressionCV()
        else:
            logistic = lm.LogisticRegression(C=C)
        return logistic

    def _make_arg_model(self, config):
        return lm.LinearRegression()


class LogisticPolynomialModel(TwoStepModel):
    def _make_sub_model(self, config):
        if config.get('cv_reg'):
            logistic = lm.LogisticRegressionCV()
        else:
            logistic = lm.LogisticRegression(C=C)
        return pl.Pipeline([('poly', pp.PolynomialFeatures(degree=config.degree)), ('logistic', logistic)])

    def _make_arg_model(self, config):
        return pl.Pipeline([('poly', pp.PolynomialFeatures(degree=config.degree)), ('linear', lm.LinearRegression(fit_intercept=False))])


class LogisticMLPModel(TwoStepModel):
    def _make_sub_model(self, config):
        if config.get('cv_reg'):
            logistic = lm.LogisticRegressionCV()
        else:
            logistic = lm.LogisticRegression(C=C)
        return pl.Pipeline([('poly', pp.PolynomialFeatures(degree=config.degree)), ('logistic', logistic)])

    def _make_arg_model(self, config):
        return nn.MLPRegressor(config.hidden_sizes, solver='lbfgs', max_iter=1000)

    def predict(self, iput, only_sub=False):
        oput = super(LogisticMLPModel, self).predict(iput, only_sub)
        if not only_sub:
            oput.arg = np.round(oput.arg)
        return oput


class TableModel(object):
    def __init__(self, config):
        """

        Args:
            config (DictTree)
        """
        self.num_sub = config.num_sub
        self.table = None
        self.const = None

    def fit(self, data):
        self.table = {self.disc(iput): (sub, tuple(arg)) for iput, sub, arg in zip(data.iput, data.oput.sub, data.oput.arg)}
        self.const = collections.Counter(self.table.values()).most_common(1)[0][0]

    def predict(self, iput):
        pred = [self.table.get(self.disc(i), self.const) for i in iput]
        pred_sub = np.asarray([p[0] for p in pred])
        pred_arg = np.asarray([p[1] for p in pred])
        return DictTree(sub=pred_sub, arg=pred_arg)

    @staticmethod
    def disc(l):
        return tuple(int(np.round(x)) for x in l)
