# Reflections

**reflection.md (150–200 words) answering:**
* Biggest surprise about Azure tooling
* A challenge you hit and fix applied
* When to choose Data Factory vs. 
* Databricks notebooks.

**A) Biggest surprise about Azure tooling.**
1) Databricks from one model can mount another model blob storage if blob have share access and SSL key.
2) Creating new compute engine for each DataBricks under resource goup is not cost effective.
3) Azure tool set up provide variety of setup and it publish/deploy untill all the checks are passed. Even with one error we can't proceed further.
   For example: If the path of file is not proper then next step is not allowed.
5) Endless possibility of multiple system setup with paying cost to hardware as minimum as $5 USD. Pure SAAS service which is pay per use.
6) It's serverless, we can work on synapse any time we need and share it when needed. Like moving synapse from one resource group to another group and then use with exising configuration.
7) We can clone the existing pipeline, Datasets, which actually helps to avoid setup time for new when the change is very minor.

**B) A challenge you hit and fix applied**

Among the several challanges faced, few are fixed by reading tooltip, navigating back to previous sections, etc, I like to talk about two challanges which helped me to learn additional informations.
   - First: How to add a dynamic date in the pipeline ? Adding these features at the creation level was not working properly. Later going to Destnation_datset_properties, updating the file name is very easy through Add Dynamic Content feature.[Alt+Shift+D] this feature is available in several microsoft applications as well like PowerApps. This method is easy fix for UI users.
   - Second: Where to find the data where it shows success in Json format. Searching this option helped me to understand one feature about Azure pipeline architecture, the success status is not passed back to source and it's recorded in the output section of pipeline run activity. From here i've captured the json data to show it's success.


**C) When to choose Data Factory vs. Databricks notebooks.**
Even though both are available in the Azure ADF provide tool with low code/ no code options to setup ETL pipelines, copy data from one source to another, we can schedule triggers, helps to connect different Azure services like SQL to Blob storage, blob to blob storage etc. It's more suitable if user wants to use these options with very little coding experience
DataBricks provide option to use different compute engine for working on Fast, Advanced, Big Data for Data Science and ML, suitable for developers, datascience users who know how to code and connect to different azure services for Big data services.

**For example**
If a project need to use both the technology then ADF can be used to trigger the Databrick notebook, databrick then perfomes complex activity, then use the optput from data bricks triggers ADF to store or copy the data to another storage.
