## Part 1: Warmup -- Check for Understanding

### Cloud Concepts

### Cloud Concepts Question 1

Core economic model is 'pay-as-you-go', so it saves money compared to buying and maintaining servers. With cloud you pay for what you use and you're not responsible for maintenance.

### Cloud Concepts Question 2

Vertical scaling - increasing performance within a single server/machine.
Horizontal scaling - increasing the number of servers.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch. - Needs horizontal scaling since the number of users increases.
A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM. - Needs vertical scaling, to increase resources for a single job.
A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines. - Horizontal, as it needs to be spread across multiple machines.

### Cloud Concepts Question 3

Gmail - SaaS
Azure Virtual Machines - Iaas
Azure App Service - PaaS
AWS S3 (Simple Storage Service) - IaaS
GitHub Codespaces - PaaS
Snowflake - PaaS

SaaS - You get to use an application. No need to build anything, just be a regular user.
Example: Gmail. A user can changes tons of settings but not how the software works.
IaaS - nothing is prebuilt, you get resources to build your own platforms and applications. Example: Azure Virtual Machines. A developer has to set up everything from scratch on the provided 'empty' machines.
PaaS - you are given a maintained infrastructure and you build on top of it. A developer is responsible for writing code.

### Cloud Concepts Question 4

Data platforms run on top of a cloud. They are already configured to perform data related tasks, unlike cloud providers who require lots of configuration.
So you give up some flexibility but gain more options to work with data - storage, pipelines etc. Another loss is that a data platform owns the pipeline and a user is dependent on it if they change conditions of use.

### Cloud Concepts Question 5

If a dataset can be managed on a single machine and doesn't require scaling, cloud is not a good choice. Also, local machine is preferable when setting up an initial prototype.

## Azure Basics

### Azure Basics Question 1

A subscription belongs to one billing account that manages the resources. All CTD users share the same subscription. A resource group can be allocated to a particular user within that organization. It is similar to a directory.

### Azure Basics Question 2

By default, there's no session memory. Nothing is stored outside a single session. Connecting the shell to a file share helps make it persistent, i.e. maintain memory between sessions.

### Azure Basics Question 3

Public key gets uploaded to remote systems; private key stays on the local machine. Sharing a public key is safe because no access will be granted with public key only - matching to a private key needs to happen for that.

### Azure Basics Question 4

{
"environmentName": "AzureCloud",
"homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
"id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
"isDefault": true,
"managedByTenants": [],
"name": "CTD Nonprofit Sponsorship",
"state": "Enabled",
"tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
"user": {
"cloudShellID": true,
"name": "live.com#izumrud@gmail.com",
"type": "user"
}
}

--output table returns the same info as a table and not a json
