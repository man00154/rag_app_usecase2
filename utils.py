import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

# ---------------- PDF & HTML LINKS ---------------- #
PDF_LINKS = {
    # Dell
    "PowerEdge Rack Servers Quick Reference Guide":
        "https://i.dell.com/sites/csdocuments/Product_Docs/en/Dell-EMC-PowerEdge-Rack-Servers-Quick-Reference-Guide.pdf",
    "PowerEdge R660xs Technical Guide":
        "https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-r660xs-technical-guide.pdf",
    "PowerEdge R740/R740xd Technical Guide":
        "https://i.dell.com/sites/csdocuments/shared-content_data-sheets_documents/en/aa/poweredge_r740_r740xd_technical_guide.pdf",
    "OpenManage Server Administrator v9.5 User’s Guide":
        "https://dl.dell.com/topicspdf/openmanage-server-administrator-v95_users-guide_en-us.pdf",
    "System Configuration Profiles Reference Guide":
        "https://dl.dell.com/manuals/common/dellemc-server-config-profile-refguide.pdf",

    # IBM
    "Power Systems Virtual Server Guide for IBM i":
        "https://www.redbooks.ibm.com/redbooks/pdfs/sg248513.pdf",
    "SPSS Statistics Server Administrator’s Guide":
        "https://www.ibm.com/docs/SSLVMB_28.0.0/pdf/IBM_SPSS_Statistics_Server_Administrator_Guide.pdf",
    "HTTP Server v6 User’s Guide":
        "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
    "Storage Protect PDF Documentation Index":
        "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files",

    # Cisco
    "Enterprise Campus Infrastructure Design Guide":
        "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
    "IT Wireless LAN Design Guide":
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
    "IT IP Addressing Best Practices":
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
    "Network Registrar 7.2 User Guide":
        "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",

    # Juniper
    "Junos Overview":
        "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
    "Junos OS Network Management Administration Guide":
        "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
    "Junos Space Network Management Security Policy":
        "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",

    # Fortinet (FortiGate)
    "FortiOS 5.6 Firewall Handbook":
        "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
    "FortiWeb 6.0.7 Administration Guide":
        "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
    "FortiGate-200 Administration Guide":
        "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
    "FortiGate Next-Gen Firewall Common Criteria Handbook":
        "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf"
}

HTML_LINKS = {
    "Dell EUC Overview": "https://www.dell.com/en-us/lp/dt/end-user-computing",
    "Nutanix EUC Solutions": "https://www.nutanix.com/solutions/end-user-computing",
    "EUC Score Toolset Documentation": "https://eucscore.com/docs/tools.html",
    "Apparity EUC Governance Docs Repository": "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
    "IBM Storage Protect PDF Documentation Index": "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files"
}

def ensure_data():
    os.makedirs("sample_data", exist_ok=True)
    os.makedirs("html_data", exist_ok=True)

    # Download PDFs
    for name, url in PDF_LINKS.items():
        path = os.path.join("sample_data", name.replace(" ", "_") + ".pdf")
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed PDF: {name} — {e}")

    # Download HTML and save as text
    for name, url in HTML_LINKS.items():
        path = os.path.join("html_data", name.replace(" ", "_") + ".txt")
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"Failed HTML: {name} — {e}")

def load_pdfs_from_folder(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(folder, file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return docs

def load_html_from_folder(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def create_vectorstore(pdf_docs, html_docs, store_path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(pdf_docs + html_docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(store_path, exist_ok=True)
    vectorstore.save_local(store_path)
    with open(os.path.join(store_path, "index.pkl"), "wb") as f:
        pickle.dump(split_docs, f)

def load_vectorstore(store_path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(store_path, embeddings)
