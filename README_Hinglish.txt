RR AUCTION STRATEGY ROOM - RUNNABLE PROJECT

QUICK START (dashboard only)
1. Extract the ZIP to a folder on your laptop.
2. Double-click Open_Dashboard.bat, or open index.html in Edge/Chrome.
3. The dashboard is a standalone HTML file. It does not need Python or an internet connection to display. Internet is only needed if you click source links; custom fonts fall back to system fonts offline.
4. The companion PDF is RR_Auction_Top10_Submission_Report.pdf.

CONTENTS
- index.html: complete interactive RR strategy dashboard; data and dashboard code are embedded in this file.
- RR_Auction_Top10_Submission_Report.pdf: form-ready shortlist and strategy summary.
- reference/form_shortlists.json: the exact ranked batter and bowler names used in the PDF.
- rr_scout/: supplied Python scouting pipeline, source files, dependencies and the original extracted Data Pack.
- Run_Scout_Pipeline.bat: optional Windows launcher to install dependencies, rerun the scouting pipeline and open its Streamlit app.

OPTIONAL: RE-RUN THE PYTHON SCOUTING PIPELINE
Requires Python 3.10+ and internet on the first run to install packages. Double-click Run_Scout_Pipeline.bat. It creates a local virtual environment under rr_scout/.venv, installs rr_scout/requirements.txt, runs run_all.py, and then starts the Streamlit scouting app. Generated scouting CSVs are written under rr_scout/outputs/.

IMPORTANT SCOPE NOTE
The standalone dashboard and PDF are the completed RR auction strategy deliverables. The optional rr_scout pipeline is the supplied Python scouting project and writes its own scouting CSVs; it does not rebuild or overwrite the strategy dashboard/PDF. The Top 10 is ordered by RR role needs, with dataset scores, pressure/death indicators and sample cautions shown. Availability, registration, NOC and fitness still need final confirmation before any real auction decision.
