from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError

class VMUtil(): 
    def __init__(self, configs): 
        """
        :param configs: experiment configurations
        :type configs: Configuration
        """
        self.configs = configs
        # 修复VM_URL为None的问题
        if configs.VM_URL:
            self.vm_url = "http://" + configs.VM_URL
        else:
            self.vm_url = None
        
    def http_request(self, url, data=None):
        """
        Sends a http request with to an url
        :param url: target url
        :type url: string
        :param data: data for a post request
        :type data: string
        return urllib.Request, object
        """
        try:
            request = Request(url, data=data.encode('ascii', 'ignore')) if data else Request(url)
            response = urlopen(request)
            return request, response
        except HTTPError as e: 
            return None, None
            print("ERROR: {}".format(e))

    def get_data_by(self, name):
        """
        Returns all entries of a metrics 
        :param name: metrics __name__
        :type name: string
        return string
        """
        if self.vm_url is None:
            return ""
        url = self.vm_url +"/api/v1/query?query=%s{}[2y]"% (name)
        request, response = self.http_request(url)
        if response:
            return response.read().decode("utf-8")
        return ""
        
    def push_data(self, data):
        """
        Push data to Victoria Metrics Database
        """
        if self.vm_url is None:
            return
        url = self.vm_url + "/write?precision=s"
        try:
            request, response = self.http_request(url, data=data)
        except HTTPError as e: 
            print("ERROR: {}".format(e))
            
    def delete_old_metrics(self, name, metrics, labels=None):
        if self.vm_url is None:
            return
        target_url = self.vm_url + "/api/v1/admin/tsdb/delete_series?"
        for metric in metrics:
            print("Delete old metrics from {}_{} with {}".format(name, metric, labels))
            data = "match[]={__name__='"+ name +"_" + metric + "'," + labels + "}" if labels else "match[]={__name__='"+ name + "_" + metric + "'}" 
            try:
                request = Request(target_url, data=data.encode('ascii', 'ignore'))
                response = urlopen(request)
            except HTTPError as e: 
                print("ERROR: {}".format(e))