import streamlit as st
import requests
import pandas as pd
import plotly_express as px
import time
import datetime
import matplotlib.pyplot as plt
from digi_plotfuncs import *
from data_call import *

# Set page in wide mode
st.set_page_config(layout="wide")
# Set plotting style
plt.style.use("fivethirtyeight")

st.title("Digi Analysis")

last_call = pd.read_csv("app_data/last_call_date.csv", parse_dates=["last_call"])
st.write(f'Data up to date as of: {last_call.last_call.iloc[-1].strftime("%d-%m-%Y")}')
minutes = round((datetime.datetime.now() - last_call.last_call.iloc[-1]).seconds/60)
days = (datetime.datetime.now() - last_call.last_call.iloc[-1]).days
st.write(f"{days} days and {minutes} minutes ago!")

# data = st.sidebar.button("Update Data")
# if data:
df_account, df_profile, df_family, df_task_definition, df_task_execution = call_data()
print(df_account.shape)
# Else call in from .csv





view = st.sidebar.radio("Select a tool to view / call in data", ["Back End", "Front End", "Data Call"])

st.sidebar.color_picker("color")

backend = st.sidebar.button("backend")
frontend = st.sidebar.button("frontend")

def plotter(dataframe, plot_stat, df_type, width=250):
    # fig = px.line(plot_df, x=metric_sel, y=points_type,
    #                  animation_frame="round", animation_group="code",
    #                  range_x=[min_x, max_x], range_y=[0, max_y], size="sel_size",
    #                  hover_name="web_name", color="plot_color", opacity=0.9,
    #                  hover_data={'now_cost': True, 'round': False, "gw_points": True, "total_points": True,
    #                              'minutes': True, 'plot_color': False, 'sel_size': False},
    #                  width=1000, height=550)
    plt.style.use("fivethirtyeight")
    fig = px.line(data_frame=dataframe, x=plot_stat, y=dataframe.index,
                  width=width, height=400,

                  title=f"{df_type}"
                  )

    # fig.update_layout(title=f"Account Growth: {df_type}")
    fig.update_layout(yaxis_title="Number of Accounts", xaxis_title="Days App Live")


    return fig

if view == "Back End":
    statSelect = st.radio("Select Statistics", ["account", "family", "profile"], horizontal=True)

    data_collection = requests.get(f"https://int1.digitheapp.com:5400/report-account/get-{statSelect}-collection",
                                      verify=False).json()
    df = pd.DataFrame.from_dict(data_collection)
    df["creationTime"] = pd.to_datetime(df.creationTime)
    df["creationTime"] = df.creationTime.dt.strftime('%Y-%m-%d %H:%M:%S')
    df["creationTime"] = pd.to_datetime(df.creationTime)
    df["dayOfYear"] = df.creationTime.dt.dayofyear
    df["daysLive"] = df.dayOfYear - 47

    df = df.dropna(subset=["creationTime"], axis=0)
    print(df.columns)

    # Split by child and adult accounts
    if statSelect == "account":
        df_child = df.query("accountType == 'CHILD'").reset_index(drop=True).sort_values('creationTime')
        df_adult = df.query("accountType == 'ADULT'").reset_index(drop=True).sort_values('creationTime')

        df= df.sort_values("creationTime")

        st.subheader(statSelect.title() + " Adoption Rate")

        sel_1, sel_2, sel_3 = st.columns([3, 2, 2])

        with sel_1:
            st.plotly_chart(plotter(df, plot_stat="daysLive", df_type="All Accounts", width=400))

        with sel_2:
            st.plotly_chart(plotter(df_child, plot_stat="daysLive", df_type="Child Accounts"))

        with sel_3:
            st.plotly_chart(plotter(df_adult, plot_stat="daysLive", df_type="Adult Accounts"))



        st.subheader("Tasks Summary")
        st.write("Each marker represents a single task record!")
        account_tasks_df = build_tasks_df(df_account_dump=df_account, df_task_execution_dump=df_task_execution,
                                          df_task_definition_dump=df_task_definition
                                          )

        task_col_1, task_col_2 = st.columns([1,1])
        with task_col_1:
            st.plotly_chart(plotly_strip_plots(data=account_tasks_df))
        with task_col_2:
            st.plotly_chart(plotly_strip_plots(data=account_tasks_df, y="taskDuration",
                                               title="Task Duration per Child / Task Type", ylabel="Task Duration"))

        task_col_3, task_col_4 = st.columns([1,1])
        with task_col_3:
            st.plotly_chart(plotly_strip_plots(data=account_tasks_df, x="fam_ref", y="taskDuration",
                                               title="Task Duration for Family by Child ", xlabel="Family Name",
                                               ylabel="Task Duration(H)", legend=True,
                                               color="child_ref", stripmode="overlay"))
        with task_col_4:
            st.plotly_chart(plotly_strip_plots(data=account_tasks_df, x="weekOfYear", y="taskDuration", color="fam_ref",
                                              title="Task Duration per Family - Weekly Basis", xlabel="Week of Year",
                                              ylabel="Task Duration", legend=True, stripmode="overlay"))


        st.subheader(statSelect.title() + " Operating Systems")

        # Get new features relating to operating systems
        df["devicesLinked"] = [len(row.devices) for ind, row in df.iterrows()]
        devicesInfo = [row.devices if len(row.devices) > 0 else None for ind, row in df.iterrows()]
        df["devicesInfo"] = devicesInfo


        def get_device_type(row):

            devices = []

            if row.devicesInfo is None:
                return "No Device"
            else:
                list_of_types = [x["type"] for x in row.devicesInfo]
                return ", ".join(list_of_types)


        df["deviceOS"] = df.apply(get_device_type, axis='columns')

        os = df.groupby(["deviceOS", "status"]).count()[["_id"]].rename(columns={"_id": "osCount"})
        osCount = df.groupby("deviceOS").count()[["_id"]].rename(columns={"_id": "osCount"})

        osCount_ = osCount.reset_index().rename(columns={"osCount": "osTotal"})
        os_ = os.reset_index()

        osA = os_.merge(osCount_, on="deviceOS").set_index(["deviceOS", "status"])

        plot_os = osA[["osCount"]].unstack()
        plot_os.columns = ["Active", "Inactive", "Inactive_Pending", "Pending"]

        plot_os = plot_os.fillna(0).reset_index()

        fig1 = px.bar(plot_os, x=plot_os.index, y=["Active", "Inactive", "Inactive_Pending", "Pending"],
                     barmode='group'
                     )  # .update_traces(width=0.2)

        fig1.update_layout(
            xaxis=dict(
                tickangle=-35,
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=plot_os.deviceOS.tolist()
            )
        )

        fig1.update_layout(
            xaxis_title="Operating System",
            yaxis_title="Status Count",

            title=f"Status of All Linked Operating Systems",
            # title= f"Requested Metrics: volatility < {round(vol,3)} over {timeperiod} days",
            title_font_color="#003865"

        )

        st.plotly_chart(fig1)

    elif statSelect == "family":

        st.selectbox("Select a family id", )

    st.subheader("Data Visual Ideas")
    st.write("Tasks Started / Completed in last n days")
    st.write("Task completion time per child")
    st.write("Membership Expiries in next 1,2,7,14,30 days")
    st.write("Operating Systems Linked")
    st.write("Expected time vs actual time to complete tasks")


elif view=="Data Call":
    st.title("Data Call in Page")

    if st.button("update data"):
        with st.spinner('Wait for it...'):
            st.write("Awaiting security permissions... No call available at this time")
            # data_call()
            time.sleep(5)



else:
    statSelect = st.radio("Select Profile Statistics", ["tasks", "history", "status"], horizontal=True)
    sel_1, sel_2, sel_3 = st.columns(3)

