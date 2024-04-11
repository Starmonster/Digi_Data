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
st.sidebar.subheader("LAST DATA CALL:")
st.sidebar.write(f'Data up to date as of: {last_call.last_call.iloc[-1].strftime("%d-%m-%Y")}')
minutes = round((datetime.datetime.now() - last_call.last_call.iloc[-1]).seconds/60)
days = (datetime.datetime.now() - last_call.last_call.iloc[-1]).days
st.sidebar.write(f"{days} days and {minutes} minutes ago!")

# data = st.sidebar.button("Update Data")
# if data:
df_account, df_profile, df_family, df_task_definition, df_task_execution = call_data()
print(df_account.shape)
# Else call in from .csv





view = st.sidebar.radio("Select a tool to view / call in data", ["Back End", "Front End", "Data Call", "Synthetic"])

# st.sidebar.color_picker("color")
#
# backend = st.sidebar.button("backend")
# frontend = st.sidebar.button("frontend")


def plotter(dataframe, plot_stat, df_type, width=250):

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



elif view=="Synthetic":
    sel1, sel2, sel3 = st.columns(3)
    with sel1:
        time_period = st.selectbox("Select a Time Period", ["Week", "Month"])

    st.plotly_chart(generate_plot_dense_data(time_period=time_period
    ))


elif view == "Front End":
    statSelect = st.radio("Select Profile Statistics", ["family"], horizontal=True)
    data_collection = requests.get(f"https://int1.digitheapp.com:5400/report-account/get-{statSelect}-collection",
                                   verify=False).json()
    df = pd.DataFrame.from_dict(data_collection)
    sel1, sel2, sel3 = st.columns(3)
    with sel1:
        fam_select = st.selectbox("Select Family Id", df._id.unique())

    sel_1, sel_2 = st.columns([2, 1])
    with sel_1:

        account_tasks_df = build_tasks_df(df_account_dump=df_account, df_task_execution_dump=df_task_execution,
                                          df_task_definition_dump=df_task_definition
                                          )

        unique_fam_ids = account_tasks_df.familyId.unique()
        fam_ref_df = pd.DataFrame(
            {'familyId': unique_fam_ids, 'fam_ref': [x for x in range(1, len(unique_fam_ids) + 1)]})

        fam_tasks_df = account_tasks_df.query(f"familyId=='{fam_select}'")
        fam_tasks_date = fam_tasks_df.groupby("creationDate").count().iloc[:, 1:2]
        fam_tasks_date = fam_tasks_date.reset_index(drop=False).rename(columns={"createdBy": "taskCount"})
        fam_tasks_date["cumulativeTasks"] = fam_tasks_date.taskCount.cumsum()
        fam_tasks_date["cumulativeTasksComp"] = round(fam_tasks_date.cumulativeTasks * random.uniform(0.3, 0.8))

        # Get the mean task duration / completion per family
        family_mean_duration = (account_tasks_df[["familyId", "taskDuration"]]
                                .groupby("familyId")
                                .mean()
                                .reset_index()
                                .rename(columns={"taskDuration": "meanDuration(hrs)"})
                                ).round(3)

        task_count_by_family = (account_tasks_df
                                .groupby("familyId")
                                .count()
                                .iloc[:, 1]
                                .to_frame()
                                .rename(columns={"createdBy": "tasksCount"}))
        task_count_by_family = task_count_by_family.reset_index()

        accounts = df_account[["_id", "profileId", "familyId", "accountType", "creationTime"]]
        account_types = accounts.groupby(["familyId", "accountType"]).count()  # .reset_index()
        account_types = account_types.iloc[:, 1].to_frame().rename(columns={"profileId": "typeCount"}).reset_index()
        # First get only the number of childred per family
        account_types_child = account_types.query("accountType=='CHILD'").rename(columns={"typeCount": "childCount"})
        # account_types_child
        # Then merge with task count
        master_family_task = task_count_by_family.merge(account_types_child[["familyId", "childCount"]], on="familyId")
        # Now merge with mean duration
        master_family_task = master_family_task.merge(family_mean_duration, on="familyId")

        master_family_task = master_family_task.merge(fam_ref_df, on="familyId").sort_values("fam_ref")

        family_details = master_family_task.query(f"familyId=='{fam_select}'")

        info1, info2, info3 = st.columns(3)
        with info1:
            variable_output = family_details.tasksCount.iloc[0]
            html_str = f"""
                                <style>
                                p.a {{
                                  font: bold {50}px Courier;
                                  color: DarkGrey;
                                }}
                                </style>
                                <p class="a">{variable_output}</p>
                                """
            st.subheader("Number of tasks created:")
            st.markdown(html_str, unsafe_allow_html=True)
        with info2:
            variable_output = family_details.childCount.iloc[0]
            # font_size = st.slider("Enter a font size", 1, 300, value=30)
            html_str = f"""
                                <style>
                                p.a {{
                                  font: bold {50}px Courier;
                                }}
                                </style>
                                <p class="a">{variable_output}</p>
                                """
            st.subheader("Number of active children:")
            st.markdown(html_str, unsafe_allow_html=True)
        with info3:
            variable_output = family_details['meanDuration(hrs)'].iloc[0]
            html_str = f"""
                                <style>
                                p.a {{
                                  font: bold {50}px Courier;
                                }}
                                </style>
                                <p class="a">{variable_output}</p>
                                """
            st.subheader("Average task duration:")
            st.markdown(html_str, unsafe_allow_html=True)

        # st.write(f"Number of tasks created: {family_details.tasksCount[0]}")
        # st.write(f"Number of active children: {family_details.childCount[0]}")
        # st.write(f"Average task duration: {family_details['meanDuration(hrs)'][0]}")

        # sns.set_style('darkgrid')

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle(f"Task counts for family {fam_select}", fontsize=18)
            ax.bar(fam_tasks_date.creationDate, fam_tasks_date.taskCount, alpha=0.5, label="Tasks Created by Day")
            ticks = pd.date_range(fam_tasks_date.creationDate.iloc[0], fam_tasks_date.creationDate.iloc[-1], 6)

            ax.set_xticks(ticks.date)
            ax.set_xticklabels(labels=ticks.date, rotation=45)
            ax.set_xlabel("Date", fontsize=13)
            ax.set_ylabel("Tasks Created", fontsize=13)

            ax1 = ax.twinx()
            ax1.plot(fam_tasks_date.creationDate, fam_tasks_date.cumulativeTasks, color="orange",
                     linewidth=3.2, label="Cumulative Tasks Created")
            ax1.plot(fam_tasks_date.creationDate, fam_tasks_date.cumulativeTasksComp, color="red",
                     linewidth=3.2, label="Cumulative Tasks Completed")
            ax1.set_ylabel("Cumulative Tasks", fontsize=13)

            ax1.fill_between(fam_tasks_date.creationDate, y1=fam_tasks_date.cumulativeTasks,
                             y2=fam_tasks_date.cumulativeTasksComp, cmap="viridis", alpha=0.5, color="orange")

            #     im = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto',
            #                extent=[*ax.get_xlim(), *ax.get_ylim()], zorder=10)

            fig.legend(bbox_to_anchor=(0.55, 0.85), fontsize=10)

            st.pyplot(fig)
            # fig.legend(l)
        except:
            print("No Family Tasks")
            plt.plot([2, 4], [2, 4])
            fig.suptitle("NO DATA FOUND FOR THIS FAMILY")
            st.pyplot(fig)



