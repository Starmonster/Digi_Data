# Functions

import streamlit as st
import requests
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from sshtunnel import SSHTunnelForwarder
import pymongo

########### DATA CALL FUNCTIONS #######
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

def call_db(database="digi_account", address=28118, collections=[]):
    MONGO_HOST = '35.178.138.67'
    SERVER_USER   = 'ubuntu'
    PRIVATE_KEY ='awsdigikey_int_01.pem'

    MONGO_DB = database

    # define ssh tunnel
    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_username=SERVER_USER,
        ssh_pkey=PRIVATE_KEY,
        remote_bind_address=('127.0.0.1', 28118)
    )
    # start ssh tunnel
    server.start()
    connection = pymongo.MongoClient('127.0.0.1', server.local_bind_port)
    db = connection[MONGO_DB]
    # server.close()
    return db, server


def generate_tasks_df(tasks_data, child_select="", family_select="", task_view="Total"):
    """
    Builds the task dataframe based on user request for total, child or family query
    """
    tasks_data["creationDate"] = pd.to_datetime(tasks_data.creationDatetime.dt.date)

    if task_view == "Child":  # We don't have family build in yet
        # Query by child selected before compiling the data.
        tasks_df = (tasks_data
                    .query(f"childrenIds=='{child_select}'")
                    .groupby(["creationDate"])
                    .count()
                    .iloc[:, 0]
                    .to_frame()
                    .rename(columns={"_id": "taskCount"})
                    )
    elif task_view == "Family":
        tasks_df = (tasks_data
                    .query(f"familyId=='{family_select}'")
                    .groupby("creationDate")
                    .count()
                    .iloc[:, 0]
                    .to_frame()
                    .rename(columns={"_id": "taskCount"})
                    )
    else:
        tasks_df = (tasks_data
                    .groupby("creationDate")
                    .count()
                    .iloc[:, 0]
                    .to_frame()
                    .rename(columns={"_id": "taskCount"})
                    )

    tasks_df = tasks_df.reset_index()

    # Get all the days in the date range availle
    full_period = pd.date_range(tasks_df.creationDate.min(), tasks_df.creationDate.max())
    full_period_df = pd.DataFrame(full_period, columns=["creationDate"])

    tasks_df = full_period_df.merge(tasks_df,
                                    on="creationDate", how="left").fillna(0)

    tasks_df["weekOfYear"] = tasks_df.creationDate.dt.strftime('%U')

    tasks_df["inceptionDate"] = pd.to_datetime("2023-03-11")
    tasks_df["daysLive"] = (tasks_df.creationDate - tasks_df.inceptionDate).astype(str)
    tasks_df["daysLive"] = tasks_df.daysLive.str.replace(" days", "").astype(int) + 1
    tasks_df["weeksLive"] = (round(tasks_df.daysLive / 7) + 1).astype(int)
    tasks_df["month"] = tasks_df.creationDate.dt.month_name()
    tasks_df["monthsLive"] = (round(tasks_df.daysLive / 30) + 1).astype(int)
    tasks_df["year"] = tasks_df.creationDate.dt.year

    return tasks_df




def plot_tasks_created(task_data, time_scale, window, task_view="Total"):
    """
    Create a visual to display task creation through a variety of different time periods and the option\
    to view average tasks created thro' requested rolling windows.

    time_scale: Days, Weeks, Months, Date (Option to add years by updating time_dict in far future :))
    window: int()

    """
    # Store feature key value pairs for plotting
    time_dict = dict({"Date": "creationDate", "Days": "daysLive", "Weeks": "weeksLive", "Months": "month"})
    if time_scale == "Weeks" or time_scale == "Months":
        plot_df = task_data.groupby(time_dict[time_scale])["taskCount"].sum().to_frame().reset_index()
        plot_df["rolling_ave"] = plot_df.taskCount.rolling(window=window, min_periods=1).mean()
        fig = go.Figure(
            go.Bar(x=plot_df[time_dict[time_scale]], y=plot_df.taskCount, name="tasks created")
        )

        fig.add_trace(
            go.Line(x=plot_df[time_dict[time_scale]], y=plot_df.rolling_ave,
                    name=f"{window} {time_scale[:-1]} rolling ave", hovertemplate="%{y}<extra>rolling ave.</extra>")
        )

    else:
        plot_df = task_data.copy()

        plot_df["rolling_ave"] = task_data.taskCount.rolling(window=window, min_periods=1).mean()

        fig = go.Figure(
            go.Bar(x=plot_df[time_dict[time_scale]], y=plot_df.taskCount, name="tasks created")
        )

        fig.add_trace(
            go.Line(x=plot_df[time_dict[time_scale]], y=plot_df.rolling_ave, name=f"{window}D rolling ave",
                    hovertemplate="%{y}<extra>rolling ave</extra>")
        )

    if task_view != "Total":
        title_add = " - " + task_view
    else:
        title_add = ""
    fig.update_layout(
        legend=dict(
            orientation="h"),
        template="plotly_white",
        xaxis_title=dict(text=time_scale, font_size=18),
        yaxis_title=dict(text="Number of Task Created", font_size=18),
        hovermode='x',
        title=dict(text=f"Tasks Created {title_add}", font_size=24),
        width=900,
        height=550
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0),
    )

    fig.update_xaxes(showgrid=False)
    #     fig.update_yaxes(title_font=dict(size=18))

    return fig


def pie_plot(tasks_data, child_select="", family_select="", task_view="Total"):

    if task_view == "Child":
        tasks_df = tasks_data.query(f"childrenIds=='{child_select}'")
        tasks_df_grouped = (tasks_df
                               .groupby("task_type")
                               .count()
                               .iloc[:, 0]
                               .to_frame()
                               .rename(columns={"_id": "taskCount"})
                              )
    elif task_view == "Family":
        tasks_df = tasks_data.query(f"familyId=='{family_select}'")
        tasks_df_grouped = (tasks_df
                            .groupby("task_type")
                            .count()
                            .iloc[:, 0]
                            .to_frame()
                            .rename(columns={"_id": "taskCount"})
                            )

    elif task_view == "Total": # Total
        tasks_df = tasks_data.copy()
        tasks_df_grouped = (tasks_df
                            .groupby("task_type")
                            .count()
                            .iloc[:, 0]
                            .to_frame()
                            .rename(columns={"_id": "taskCount"})
                            )

    # colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig2 = go.Figure(data=[go.Pie(labels=tasks_df_grouped.index,
                                 values=tasks_df_grouped.taskCount)])
    fig2.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(line=dict(color='#000000', width=2)))
    fig2.update_layout(width=300, height=300,margin=dict(l=5, r=0, t=40, b=5)
                       )
#     fig2.show()
    return fig2

def plotter(dataframe, plot_stat, period_select, df_type, width=350, rolling=False):

    # Plot account adoption
    fig1 = go.Figure(
        go.Line(x=dataframe[plot_stat], y=dataframe.newAccountAve,
                name=f"Rolling {period_select} Day Sum")
    )

    # Add the rolling period plot
    fig1.add_trace(
        go.Line(x=dataframe[plot_stat], y=dataframe.totalAccounts, name="Total New Accounts")

    )

    fig1.update_layout(
        legend=dict(
            orientation="h"),
        title=f"New{df_type}Accounts",
        width=width
    )

    return fig1




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
                      xlabel="Child Name", ylabel="Under / Over Target", stripmode="group",
                      ):
    fig = px.strip(data, x=x, y=y, color=color, width=500, height=350, stripmode=stripmode
                   )

    fig.update_layout(showlegend=legend)
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,

        title=title
    )

    #  fig.show()

    return fig


def status_pie(new_accounts_df, task_view, width=350):
    # task_view = input("Do you want to view by child, adult or total?")
    if task_view.lower() != "total":
        new_accounts_child = new_accounts_df.query(f"accountType=='{task_view.upper()}'")
        accounts_status_grouped = (new_accounts_child
                                   .groupby("status")
                                   .count()
                                   .iloc[:, 0]
                                   .to_frame()
                                   .rename(columns={"creationDate": "numAccounts"})
                                   )

    else:
        accounts_status_grouped = (new_accounts_df
                                   .groupby("status")
                                   .count()
                                   .iloc[:, 0]
                                   .to_frame()
                                   .rename(columns={"creationDate": "numAccounts"})
                                   .drop(0, axis=0)
                                   )

    status_fig = go.Figure(data=[go.Pie(labels=accounts_status_grouped.index,
                                        values=accounts_status_grouped.numAccounts)])
    status_fig.update_traces(hoverinfo='label+percent', textinfo="value")
    status_fig.update_layout(width=width, height=400, title=task_view.title())
    # status_fig.show()

    return status_fig
###### SYNTHETIC DATA ####

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