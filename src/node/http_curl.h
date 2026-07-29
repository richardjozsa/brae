#pragma once
// The libcurl-backed HttpClient. Separate header so everything above it can be built and tested without curl.
#include "http.h"

#include <string>

namespace brae::node {

class CurlHttp : public HttpClient
{
public:
    CurlHttp();
    HttpResponse send(const HttpRequest& req) override;

    void setNodeId(std::string id) { nodeId_ = std::move(id); }
    void setUserAgent(std::string ua) { userAgent_ = std::move(ua); }

private:
    std::string nodeId_;
    std::string userAgent_ = "brae-agent/0.3.0";
};

}  // namespace brae::node
