import itertools
import json
import os
import pickle
import sys
import traceback

import flask
import flask_sqlalchemy as sql
import numpy as np
import sklearn.model_selection as ms
import sqlalchemy_utils as sau

if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models  # noqa: E402
import utils  # noqa: E402
from utils import DictTree  # noqa: E402

DEBUG = False
# logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

app_name = 'SkillHub'
app = flask.Flask(app_name)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skillhub/skillhub.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.json_encoder = DictTree.JSONEncoder
db = sql.SQLAlchemy(app)

MIN_VALID_DATA = 2
NUM_FOLDS = 20
EPSILON = 1e-6


class SubSkill(db.Model):
    __tablename__ = 'sub_skills'

    domain_name = db.Column(db.String, db.ForeignKey('agent_skills.domain_name'), primary_key=True)
    agent_name = db.Column(db.String, db.ForeignKey('agent_skills.agent_name'), primary_key=True)
    skill_name = db.Column(db.String, db.ForeignKey('agent_skills.skill_name'), primary_key=True)
    sub_skill_index = db.Column(db.Integer, primary_key=True)
    sub_skill_name = db.Column(db.String, db.ForeignKey('agent_skills.skill_name'))


class AgentSkill(db.Model):
    __tablename__ = 'agent_skills'

    domain_name = db.Column(db.String, primary_key=True)
    agent_name = db.Column(db.String, primary_key=True)
    skill_name = db.Column(db.String, primary_key=True)
    elementary = db.Column(db.Boolean, nullable=False)
    sub_skills = db.relationship(
        'AgentSkill', secondary=SubSkill.__table__,
        primaryjoin=db.and_(domain_name == SubSkill.domain_name, agent_name == SubSkill.agent_name, skill_name == SubSkill.skill_name),
        secondaryjoin=db.and_(domain_name == SubSkill.domain_name, agent_name == SubSkill.agent_name, skill_name == SubSkill.sub_skill_name),
        order_by=SubSkill.sub_skill_index)
    min_valid_data = db.Column(db.Integer, nullable=False)
    sub_arg_accuracy = db.Column(sau.ScalarListType(float), nullable=False)
    validated = db.Column(db.Boolean)
    data = db.Column(db.PickleType)
    skill_model_id = db.Column(db.Integer, db.ForeignKey('skill_models.id'))
    skill_model = db.relationship('SkillModel', primaryjoin='AgentSkill.skill_model_id == SkillModel.id', backref='agent_skills')

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SkillModel(db.Model):
    __tablename__ = 'skill_models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    arg_in_len = db.Column(db.Integer, nullable=False)
    max_cnt = db.Column(db.Integer)
    ret_in_len = db.Column(db.Integer, nullable=False)
    num_sub = db.Column(db.Integer, nullable=False)
    arg_out_len = db.Column(db.Integer, nullable=False)
    model = db.Column(db.PickleType)


db.create_all()


@app.route('/agent/<domain_name>/<agent_name>/', methods=['DELETE', 'POST', 'PUT', 'GET'])
def agent(domain_name, agent_name):
    try:
        if flask.request.method == 'DELETE':
            res = delete(domain_name, agent_name)
        elif flask.request.method == 'POST':
            res = register(domain_name, agent_name)
        elif flask.request.method == 'PUT':
            res = train(domain_name, agent_name)
        else:  # GET
            agent_skills = AgentSkill.query.filter(AgentSkill.domain_name == domain_name, AgentSkill.agent_name == agent_name).all()
            res = [agent_skill.as_dict() for agent_skill in agent_skills]
        return flask.jsonify(res)
    except Exception, e:
        if DEBUG:
            print(traceback.format_exc())
            flask.request.environ.get('werkzeug.server.shutdown')()
        else:
            raise e


def delete(domain_name, agent_name):
    if DEBUG:
        print("delete({}, {})".format(domain_name, agent_name))
    SubSkill.query.filter(SubSkill.domain_name == domain_name).filter(SubSkill.agent_name == agent_name).delete()
    AgentSkill.query.filter(AgentSkill.domain_name == domain_name).filter(AgentSkill.agent_name == agent_name).delete()
    SkillModel.query.filter(~SkillModel.agent_skills.any()).delete(synchronize_session='fetch')
    db.session.commit()
    return DictTree(deleted=agent_name)


def register(domain_name, agent_name):
    if DEBUG:
        print("register({}, {})".format(domain_name, agent_name))
    skillset = json.loads(flask.request.data, cls=DictTree.JSONDecoder)
    res = DictTree()
    for skill_name, skill in skillset.items():
        if DEBUG:
            print("Registering {}".format(skill_name))
        skill_model = None
        if len(skill.sub_skill_names) > 0:
            skill_model = SkillModel(
                name=skill.model_name,
                arg_in_len=skill.arg_in_len,
                max_cnt=skill.get('max_cnt'),
                ret_in_len=skill.ret_in_len,
                num_sub=1 + len(skill.sub_skill_names),
                arg_out_len=skill.arg_out_len,
            )
            res[skill_name] = skill.get('validated', False)
        db.session.add(AgentSkill(
            domain_name=domain_name,
            agent_name=agent_name,
            skill_name=skill_name,
            elementary=len(skill.sub_skill_names) == 0,
            min_valid_data=skill.get('min_valid_data') or MIN_VALID_DATA,
            sub_arg_accuracy=skill.get('sub_arg_accuracy') or [EPSILON],
            validated=skill.get('validated', False),
            skill_model=skill_model,
        ))
    db.session.add_all([SubSkill(
        domain_name=domain_name,
        agent_name=agent_name,
        skill_name=skill_name,
        sub_skill_index=sub_skill_index,
        sub_skill_name=sub_skill_name,
    ) for skill_name, skill in skillset.items() for sub_skill_index, sub_skill_name in enumerate(skill.sub_skill_names)])
    db.session.commit()
    return res


def train(domain_name, agent_name):
    if DEBUG:
        print("train({}, {})".format(domain_name, agent_name))
    config = json.loads(flask.request.data, cls=DictTree.JSONDecoder)
    skill_steps = {}
    for trace in config.batch:
        for time_step in trace:
            if isinstance(time_step.info, DictTree):
                steps = time_step.info.steps
            else:
                steps = time_step.info
            for skill_step in steps:
                skill_steps.setdefault(skill_step.name, []).append(skill_step)
    res = DictTree()
    for skill_name, steps in skill_steps.items():
        agent_skill = AgentSkill.query.filter(
            AgentSkill.domain_name == domain_name).filter(
            AgentSkill.agent_name == agent_name).filter(
            AgentSkill.skill_name == skill_name).one_or_none()
        if agent_skill is None:
            raise ValueError("Agent {}/{} has no skill {}".format(domain_name, agent_name, skill_name))
        if DEBUG:
            print("Training {} with {} new steps + {} existing".format(skill_name, len(steps), len(agent_skill.data or [])))
        agent_skill.data = (agent_skill.data or []) + steps
        if len(agent_skill.data) < agent_skill.min_valid_data:
            print("Not enough data to train {}".format(skill_name))
            agent_skill.validated = False
        else:
            shared_skills = []
            for shared_skill in config.shared_skills.get(skill_name, []):
                shared_skills.append(AgentSkill.query.filter(
                    AgentSkill.domain_name == domain_name).filter(
                    AgentSkill.agent_name == shared_skill.agent_name).filter(
                    AgentSkill.skill_name == shared_skill.skill_name).one_or_none())
            shared_skill_lists = list((itertools.chain.from_iterable(
                itertools.combinations(shared_skills, cnt)
                for cnt in range(len(shared_skills), 0, -1))))
            training_list = []
            if 'validation' in config.modes:
                training_list += [('validate', shared_skills) for shared_skills in shared_skill_lists]
            if 'training' in config.modes:
                training_list += [('train', shared_skills) for shared_skills in shared_skill_lists]
            if 'independent' in config.modes:
                training_list.append(('train', []))
            for mode, shared_skills in training_list:
                shared_data = sum((shared_skill.data for shared_skill in shared_skills), [])
                if mode == 'validate':
                    if DEBUG:
                        print("Trying to validate with {} steps from {}".format(
                            len(shared_data), [(shared_skill.agent_name, shared_skill.skill_name) for shared_skill in shared_skills]))
                    shared_data = _process(agent_skill, shared_data)
                    validated = _validate(agent_skill, shared_data, config.validate, config.model_dirname)
                else:  # train
                    if DEBUG:
                        print("Trying to train with {} steps from {}".format(
                            len(shared_data), [(shared_skill.agent_name, shared_skill.skill_name) for shared_skill in shared_skills]))
                    validated = _train(agent_skill, shared_data, config.validate, config.model_dirname)
                if validated:
                    # TODO: clean up training_model once a transfer model is finalized
                    if DEBUG:
                        print("Success!!!")
                    agent_skill.validated = True
                    break
            else:
                agent_skill.validated = False
        res[skill_name] = agent_skill.validated
    db.session.commit()
    return res


def _validate(agent_skill, shared_data, validate=True, model_dirname=None):
    model = models.catalog(DictTree(
        name=agent_skill.skill_model.name,
        arg_in_len=agent_skill.skill_model.arg_in_len,
        max_cnt=agent_skill.skill_model.max_cnt,
        num_sub=agent_skill.skill_model.num_sub,
        sub_arg_accuracy=agent_skill.sub_arg_accuracy,
    ))
    model.fit(shared_data)
    if validate:
        valid_data = _process(agent_skill, agent_skill.data)
        validated = models.validate(model, valid_data, agent_skill.sub_arg_accuracy)
    else:
        validated = True
    if validated:
        agent_skill.skill_model.model = model
        if model_dirname is not None:
            try:
                os.makedirs(model_dirname)
            except OSError:
                pass
            model_fn = "{}/{}.pkl".format(model_dirname, agent_skill.skill_name)
            pickle.dump(model, open(model_fn, 'wb'), protocol=2)
    return validated


def _train(agent_skill, shared_data, validate=True, model_dirname=None):
    model = models.catalog(DictTree(
        name=agent_skill.skill_model.name,
        arg_in_len=agent_skill.skill_model.arg_in_len,
        max_cnt=agent_skill.skill_model.max_cnt,
        num_sub=agent_skill.skill_model.num_sub,
        sub_arg_accuracy=agent_skill.sub_arg_accuracy,
    ))
    if validate:
        num_folds = min(len(agent_skill.data), NUM_FOLDS)
        kf = ms.KFold(num_folds, True)
        validation = []
        for new_train_idxs, valid_idxs in kf.split(agent_skill.data):
            train_data = _process(agent_skill, [agent_skill.data[idx] for idx in new_train_idxs] + shared_data)
            valid_data = _process(agent_skill, [agent_skill.data[idx] for idx in valid_idxs])
            model.fit(train_data)
            validation.append(models.validate(model, valid_data))
        validated = models.total_validation(validation, agent_skill.sub_arg_accuracy)
    else:
        validated = True
    if validated:
        all_data = agent_skill.data
        if shared_data is not None:
            all_data += shared_data
        all_data = _process(agent_skill, all_data)
        model.fit(all_data)
        agent_skill.skill_model.model = model
        if model_dirname is not None:
            try:
                os.makedirs(model_dirname)
            except OSError:
                pass
            model_fn = "{}/{}.pkl".format(model_dirname, agent_skill.skill_name)
            pickle.dump(model, open(model_fn, 'wb'), protocol=2)
    return validated


def _process(agent_skill, data):
    # TODO: this could be more efficient
    sub_skill_names = [None] + [sub_skill.skill_name for sub_skill in agent_skill.sub_skills]
    iput = np.asarray([
        utils.pad(step.arg, agent_skill.skill_model.arg_in_len)
        + [step.cnt]
        + utils.one_hot(sub_skill_names.index(step.ret_name), agent_skill.skill_model.num_sub)
        + utils.pad(step.ret_val, agent_skill.skill_model.ret_in_len)
        for step in data])
    sub = np.asarray([sub_skill_names.index(step.sub_name) for step in data])
    arg = np.asarray([utils.pad(step.sub_arg, agent_skill.skill_model.arg_out_len) for step in data])
    return DictTree(
        len=len(data),
        iput=iput,
        oput=DictTree(
            sub=sub,
            arg=arg,
        ),
    )


if __name__ == '__main__':
    app.run(debug=DEBUG)
