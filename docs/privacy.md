# Privacy & Security Guide - AskMe Voice Assistant

## Overview

AskMe prioritizes user privacy and data security through its **offline-first architecture**. This document outlines our privacy principles, security measures, and best practices for maintaining data confidentiality.

## Core Privacy Principles

### 1. **Complete Local Processing**
- **Zero Cloud Dependency**: All AI processing happens entirely on your local device
- **No Internet Required**: Full functionality without network connectivity
- **No Data Transmission**: User conversations never leave your device

### 2. **Data Minimization**
- **No Persistent Storage**: Conversations not saved by default
- **Configurable Retention**: Optional local conversation logging with user control
- **Minimal Metadata**: Only essential operational data collected

### 3. **User Control**
- **Transparent Configuration**: All privacy settings clearly documented
- **User Choice**: Opt-in model for any data collection features
- **Easy Deletion**: Simple conversation history clearing

## Privacy Features

### Conversation Handling

#### Default Behavior
```yaml
privacy:
  store_conversations: false  # No conversation storage by default
  conversation_retention_days: 7  # If enabled, auto-delete after 7 days
  anonymize_logs: true  # Remove personal info from system logs
  telemetry_enabled: false  # No usage analytics
```

#### Optional Local Storage
If you choose to enable conversation storage:
- **Local Files Only**: Stored in `logs/conversations_YYYYMMDD.json`
- **Encryption**: Files encrypted using AES-256
- **Auto-Deletion**: Configurable retention period
- **User Access**: Full control over stored data

### Log Privacy

#### Automatic Sanitization
- **Personal Information Filtering**: Removes potential PII from logs
- **API Key Redaction**: Sensitive tokens automatically redacted
- **Configurable Filters**: Customize what information to filter

#### Log Configuration
```yaml
logging:
  level: "INFO"  # Adjust verbosity
  file: "logs/askme.log"
  anonymize_logs: true  # Enable PII filtering
```

## Security Measures

### Model Security

#### Local Model Verification
- **Checksum Validation**: All downloaded models verified for integrity
- **Signed Models**: Support for cryptographically signed model files
- **Isolated Execution**: Models run in sandboxed environment

#### Model Updates
- **Manual Control**: User controls when/if to update models
- **Verification**: New models validated before use
- **Rollback**: Ability to revert to previous model versions

### System Security

#### Network Isolation
```yaml
# Recommended firewall configuration
# Block all outbound connections from AskMe process
# Allow only local interface binding
ui:
  host: "127.0.0.1"  # Localhost only
  port: 8080
  security:
    cors_origins: ["http://127.0.0.1:3000"]  # Restrict origins
```

#### File System Security
- **Restricted Permissions**: Models and logs use minimal file permissions
- **Sandboxed Directories**: Application confined to designated folders
- **No Elevated Privileges**: Runs with standard user permissions

### Audio Privacy

#### Voice Data Handling
- **RAM Processing**: Audio processed in memory only
- **No Audio Storage**: Voice data not written to disk by default
- **Immediate Cleanup**: Audio buffers cleared after processing

#### Microphone Access
- **Explicit Permission**: Clear user consent for microphone access
- **Visual Indicators**: UI shows when microphone is active
- **Easy Disable**: One-click microphone disable option

## Data Flows

### Voice Processing Pipeline
```
Microphone → RAM Buffer → Whisper ASR → Text → LLM → Response → TTS → Speaker
     ↑                                                                    ↓
User controls                                                    No data stored
everything here                                                  (unless opted-in)
```

### Optional Data Storage
```
Conversation → Local Encryption → JSON File → Auto-Delete Timer
     ↑                ↑                ↑              ↓
User enabled    AES-256 key    Local filesystem   User configurable
```

## Compliance & Standards

### Privacy Regulations

#### GDPR Compliance
- **Data Minimization**: Only process necessary data
- **User Rights**: Full control over personal data
- **No Profiling**: No automated decision making based on personal data
- **Transparent Processing**: Clear documentation of data handling

#### CCPA Compliance
- **No Sale of Data**: Data never sold or shared with third parties
- **User Access**: Users have full access to their data
- **Deletion Rights**: Easy data deletion capabilities

### Security Standards

#### NIST Cybersecurity Framework
- **Identify**: Clear inventory of data and systems
- **Protect**: Robust access controls and encryption
- **Detect**: Monitoring for unusual activities
- **Respond**: Incident response procedures
- **Recover**: Data recovery and system restoration plans

## Configuration Guide

### Maximum Privacy Setup

```yaml
# config.yaml - Maximum privacy configuration
privacy:
  store_conversations: false
  conversation_retention_days: 0
  anonymize_logs: true
  telemetry_enabled: false

logging:
  level: "WARNING"  # Minimal logging
  file: null  # No file logging
  anonymize_logs: true

ui:
  host: "127.0.0.1"  # Local access only
  security:
    cors_origins: []  # No external origins
    api_key: "your-secure-api-key"  # Enable authentication

development:
  debug: false
  profiling: false
  mock_audio: false
```

### Secure Deployment Setup

#### Docker Deployment
```dockerfile
# Dockerfile for secure deployment
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -s /bin/bash askme

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set secure permissions
RUN chown -R askme:askme /app
USER askme

# Run with restricted permissions
WORKDIR /app
CMD ["python", "main.py"]
```

#### Network Security
```bash
# Firewall rules (example for iptables)
# Block all outbound connections from AskMe
iptables -A OUTPUT -m owner --uid-owner askme -j DROP

# Allow only local web interface
iptables -I OUTPUT -m owner --uid-owner askme -d 127.0.0.1 -p tcp --dport 8080 -j ACCEPT
```

## Threat Model

### Identified Threats

#### 1. **Data Exfiltration**
- **Risk**: Malicious code attempting to transmit data
- **Mitigation**: Network isolation, code auditing, sandboxing

#### 2. **Model Poisoning**
- **Risk**: Compromised models providing malicious responses
- **Mitigation**: Model verification, checksums, signed models

#### 3. **Local Data Access**
- **Risk**: Other applications accessing conversation logs
- **Mitigation**: File permissions, encryption, minimal storage

#### 4. **Memory Attacks**
- **Risk**: Memory dumps revealing conversation data
- **Mitigation**: Memory clearing, process isolation, encryption

### Risk Assessment Matrix

| Threat | Likelihood | Impact | Risk Level | Mitigation |
|--------|------------|--------|------------|------------|
| Network eavesdropping | Low | High | Medium | Offline operation |
| Local file access | Medium | Medium | Medium | Encryption, permissions |
| Memory extraction | Low | High | Medium | Memory clearing |
| Model tampering | Low | High | Medium | Checksums, verification |

## Best Practices

### For Users

#### 1. **System Hardening**
- Keep operating system updated
- Use disk encryption (BitLocker, FileVault, LUKS)
- Enable firewall with restrictive rules
- Regular malware scanning

#### 2. **Access Control**
- Use strong user account passwords
- Enable screen locks
- Limit physical access to device
- Regular user account auditing

#### 3. **Data Management**
- Regularly review conversation logs (if enabled)
- Clear history when appropriate
- Monitor disk usage and clean up
- Backup configuration (not conversations)

### For Developers

#### 1. **Code Security**
- Regular security audits
- Dependency vulnerability scanning
- Static code analysis
- Input validation and sanitization

#### 2. **Testing**
- Privacy-focused testing scenarios
- Security penetration testing
- Data flow verification
- Memory leak detection

## Incident Response

### Data Breach Response Plan

#### 1. **Detection**
- Monitor for unusual file system activity
- Watch for unexpected network connections
- Alert on permission changes
- Log file integrity monitoring

#### 2. **Response**
- Immediate isolation of affected systems
- Assessment of data compromise
- User notification procedures
- System remediation steps

#### 3. **Recovery**
- Clean system restoration
- Configuration review and hardening
- User re-authentication
- Monitoring enhancement

### Contact Information

For security concerns or privacy questions:
- **Email**: security@askme-project.org
- **GPG Key**: [Available on project website]
- **Security Policy**: https://github.com/askme/security-policy

## Regular Security Reviews

### Monthly Tasks
- [ ] Review access logs
- [ ] Update security configurations
- [ ] Check for software updates
- [ ] Audit user permissions

### Quarterly Tasks
- [ ] Full security audit
- [ ] Penetration testing
- [ ] Privacy impact assessment
- [ ] Documentation updates

### Annual Tasks
- [ ] Threat model review
- [ ] Security training
- [ ] Compliance audit
- [ ] Disaster recovery testing

## Conclusion

AskMe's privacy-first design ensures that your conversations remain completely private and secure. By operating entirely offline and providing granular control over data handling, we enable you to benefit from AI assistance without compromising your privacy.

Remember: **Your voice, your data, your control.**
