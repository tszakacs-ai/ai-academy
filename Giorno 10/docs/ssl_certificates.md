# SSL Certificates in Corporate Environments

If you're running this application in a corporate environment with SSL inspection (such as Zscaler), you'll need to configure the application to use your organization's CA certificates.

## Options for SSL Certificate Configuration

1. **Environment Variable (Recommended)**:
   - Set the `SSL_CERT_PATH` environment variable to the location of your certificate bundle
   - Example: `SSL_CERT_PATH=C:\path\to\your\cert-bundle.pem`

2. **Common locations**:
   - The application will automatically look for certificates in common locations:
     - Your home directory: `~/zscaler-ca-bundle.pem`
     - The `certs` folder in the project
     - System-wide certificate store

3. **Fallback mode**:
   - If no certificate is found, the application will disable SSL verification
   - A warning will be shown in the sidebar

## Troubleshooting

If you encounter SSL certificate errors:

1. Check that your certificate file exists and is accessible
2. Make sure the certificate is in PEM format
3. Verify that the certificate includes all required root and intermediate CAs
4. If using Zscaler, you may need to export the Zscaler root certificates

For Zscaler environments, you can export the certificate bundle using the Zscaler client or from your browser's trusted certificates.