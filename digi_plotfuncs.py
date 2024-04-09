# Functions

import streamlit as st
import requests
import pandas as pd
import plotly_express as px
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats

def call_data():
    # Accounts
    account_collection = requests.get("https://int1.digitheapp.com:5400/report-account/get-account-collection",
                                      verify=False).json()
    df_account_dump = pd.DataFrame.from_dict(account_collection)
    # df_account_dump.drop(['nationalPhoneNumber', 'dialCode', 'invitedParents', 'validationCodes', 'devicePaired', 'invitedCarers', '_class'], axis=1, inplace=True)

    # Profiles
    profile_collection = requests.get("https://int1.digitheapp.com:5400/report-account/get-profile-collection",
                                      verify=False).json()
    df_profile_dump = pd.DataFrame.from_dict(profile_collection)
    # df_profile_dump.drop(['_class'], axis=1, inplace=True)

    # Families
    family_collection = requests.get("https://int1.digitheapp.com:5400/report-account/get-family-collection",
                                     verify=False).json()
    df_family_dump = pd.DataFrame.from_dict(family_collection)
    df_family_dump.drop(['_class'], axis=1, inplace=True)
    # display(df_family_dump)

    # Tasks definition
    task_definition_collection = requests.get(
        "https://int1.digitheapp.com:5400/report-task/get-task-definition-collection", verify=False).json()
    df_task_definition_dump = pd.DataFrame.from_dict(task_definition_collection)
    df_task_definition_dump.drop(['_class'], axis=1, inplace=True)
    # display(df_task_definition_dump.head())

    # Tasks execution
    task_execution_collection = requests.get(
        "https://int1.digitheapp.com:5400/report-task/get-task-execution-collection", verify=False).json()
    df_task_execution_dump = pd.DataFrame.from_dict(task_execution_collection)
    df_task_execution_dump.drop(['_class'], axis=1, inplace=True)
    # display(df_task_execution_dump.head())

    return df_account_dump, df_profile_dump, df_family_dump, df_task_definition_dump, df_task_execution_dump


def build_tasks_df(df_account_dump, df_task_execution_dump, df_task_definition_dump):

    accounts = df_account_dump[["_id", "profileId", "familyId", "accountType", "creationTime"]]
    account_types = accounts.groupby(["familyId", "accountType"]).count()  # .reset_index()
    account_types = account_types.iloc[:, 1].to_frame().rename(columns={"profileId": "typeCount"}).reset_index()

    tasks = df_task_execution_dump[["taskDefinitionId", "createdBy", "childrenIds", "creationDatetime", "endedAt"]]
    # Extrac the child ID from the Ids colum
    tasks["childId"] = tasks.childrenIds.apply(lambda x: x[0])

    # Merge the tasks with the accounts - so we can see tasks by family
    account_tasks_df = tasks.merge(accounts, left_on="createdBy", right_on="_id")

    # Convert date dtypes to datetime format
    account_tasks_df["creationDatetime"] = pd.to_datetime(account_tasks_df.creationDatetime)
    account_tasks_df["endedAt"] = pd.to_datetime(account_tasks_df.endedAt)
    # Get a duration for each task created to ended
    account_tasks_df["taskDuration"] = account_tasks_df.endedAt - account_tasks_df.creationDatetime

    account_tasks_df["taskDuration"] = account_tasks_df.taskDuration.apply(lambda x: x.total_seconds() / 3600)

    # Create some additional temporal features
    # Creation date
    account_tasks_df["creationDate"] = pd.to_datetime(account_tasks_df.creationDatetime.dt.date)
    # Specific day or week of the year
    account_tasks_df["dayOfYear"] = account_tasks_df.creationDate.apply(lambda x: x.dayofyear)
    account_tasks_df["weekOfYear"] = account_tasks_df.creationDate.apply(lambda x: x.weekofyear)

    unique_fam_ids = account_tasks_df.familyId.unique()
    fam_ref_df = pd.DataFrame({'familyId': unique_fam_ids, 'fam_ref': [x for x in range(1, len(unique_fam_ids) + 1)]})

    unique_child_ids = account_tasks_df.childId.unique()
    child_ref_df = pd.DataFrame({'childId': unique_child_ids,
                                 'child_ref': [x for x in range(1, len(unique_child_ids) + 1)]})

    # We'll need target completion and actual completion to measure how long kids beat or exceed the target
    task_valid_df = df_task_definition_dump[["_id", "validityEndDate"]]
    task_valid_df = task_valid_df.rename(columns={"_id": "taskId"})
    task_valid_df.validityEndDate = pd.to_datetime(task_valid_df["validityEndDate"])

    account_tasks_df = account_tasks_df.merge(fam_ref_df, on="familyId")
    account_tasks_df = account_tasks_df.merge(child_ref_df, on="childId")
    account_tasks_df = account_tasks_df.merge(task_valid_df, left_on="taskDefinitionId", right_on="taskId")

    account_tasks_df["targetDiff"] = account_tasks_df.validityEndDate - account_tasks_df.endedAt
    account_tasks_df["targetDiff"] = account_tasks_df.targetDiff.apply(lambda x: x.total_seconds() / 3600)

    return account_tasks_df


# fig, ax = plt.subplots()
def plotly_strip_plots(data, x="child_ref", y="targetDiff", color="taskDefinitionId",
                      title="Child Under / Over Target Completion - By Task Type", legend=False,
                      xlabel="Child Name", ylabel="Under / Over Target", stripmode="group"
                      ):
    fig = px.strip(data, x=x, y=y, color=color, width=700, height=400, stripmode=stripmode
                   )

    fig.update_layout(showlegend=legend)
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,

        title=title
    )

    #  fig.show()

    return fig


##### SYNTHETIC DATA ####
def get_density(x:np.ndarray, y:np.ndarray):
    """Get kernal density estimate for each (x, y) point."""
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    density = kernel(values)
    return density
def generate_plot_dense_data(time_period=str()):
    if time_period == "Week":
        x = np.random.normal(18, 25, 100000)
        x = np.array(random.choices(x[[z > 0 for z in x]], k=2000)) + 1
        y = ((0.8 * x + np.random.randn(2000) / 0.08) + 34) / 4.3

        df_child_tasks_comp = pd.DataFrame([x, y], index=["num_children", "tasks_completed"]).T
        df_child_tasks_comp = df_child_tasks_comp.astype(int)

        d = get_density(x, y)

        df_child_tasks_comp["density"] = d * 1000
        df_child_tasks_comp["density"] = df_child_tasks_comp.density.round(4)

    elif time_period == "Month":
        x = np.random.normal(18, 25, 100000)
        x = np.array(random.choices(x[[z > 0 for z in x]], k=2000)) + 1
        y = ((0.8 * x + np.random.randn(2000) / 0.08) + 34)

        df_child_tasks_comp = pd.DataFrame([x, y], index=["num_children", "tasks_completed"]).T
        df_child_tasks_comp = df_child_tasks_comp.astype(int)

        d = get_density(x, y)

        df_child_tasks_comp["density"] = d * 10000
        df_child_tasks_comp["density"] = df_child_tasks_comp.density.round(4)

    #         df_child_tasks_comp = generate_dense_data(time_period=time_period)
    fig = px.scatter(data_frame=df_child_tasks_comp, x=df_child_tasks_comp.num_children,
                     y=df_child_tasks_comp.tasks_completed, color=df_child_tasks_comp.density,
                     hover_name=df_child_tasks_comp.index,
                     hover_data={'num_children': True, "density": True},
                     color_continuous_scale=px.colors.sequential.Plasma, width=1000, height=400
                     )
    # fig.update_layout(showlegend=legend)
    fig.update_layout(
        xaxis_title="Number of Children in Family",
        yaxis_title=f"Tasks completed in last {time_period}",
        coloraxis_colorbar_title_text="family density",

        title=f"Family Density by Size and Tasks Completed for last {time_period}"

    )

    #     fig.show()

    return fig
