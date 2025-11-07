# AWS Bedrock Setup for Claude Sonnet 4.5

## Problem

The chat interface is showing generic fallback responses like:
```
Try: "Show CRP last 7 days", "Creatinine last 30 days"...
```

This happens because **AWS credentials are not configured**, so the application cannot authenticate with AWS Bedrock to use Claude Sonnet 4.5.

## Diagnosis

Run this test endpoint to verify Bedrock connectivity:
```bash
curl http://localhost:8000/bedrock_test | python3 -m json.tool
```

If you see `"error": "Unable to locate credentials"`, AWS credentials need to be configured.

## Solution: Configure AWS Credentials

### Option 1: AWS CLI Configuration (Recommended)

1. **Install AWS CLI** (if not already installed):
   ```bash
   pip install awscli
   ```

2. **Configure credentials**:
   ```bash
   aws configure
   ```
   
   Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region: `us-west-2` (or your Bedrock region)
   - Default output format: `json`

3. **Verify**:
   ```bash
   aws sts get-caller-identity
   ```
   
   Should show your AWS account details.

### Option 2: Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-west-2"
```

Then restart your terminal or run `source ~/.bashrc`.

### Option 3: IAM Role (for EC2/ECS)

If running on AWS infrastructure, attach an IAM role with Bedrock permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-5-*"
    }
  ]
}
```

## Enable Bedrock Access

1. **Request Model Access** in AWS Console:
   - Go to AWS Bedrock console
   - Navigate to "Model access"
   - Request access to "Claude Sonnet 4.5"
   - Wait for approval (usually instant for AWS accounts)

2. **Verify Region**:
   - Ensure Bedrock is available in your region
   - Check `.env.local`: `BEDROCK_REGION=us-west-2`

## Testing

After configuring credentials:

1. **Test Bedrock connectivity**:
   ```bash
   curl http://localhost:8000/bedrock_test
   ```
   
   Should return:
   ```json
   {
     "bedrock_enabled": "true",
     "bedrock_model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
     "bedrock_region": "us-west-2",
     "test_call": "Hello, Bedrock is working!",
     "error": null
   }
   ```

2. **Test patient chat**:
   ```bash
   curl -X POST http://localhost:8000/patient_chat \
     -H "Content-Type: application/json" \
     -d '{
       "patient_id": "DEMO-CRP-001",
       "text": "Hello! How can you help me?",
       "days_back": 7
     }'
   ```
   
   Should return an intelligent AI response, not the generic fallback.

3. **Test in UI**:
   - Open the chat interface
   - Type any general message (e.g., "hello", "help me")
   - Should receive intelligent responses from Claude Sonnet 4.5

## Environment Configuration

The following environment variables in `.env.local` control Bedrock:

```bash
BEDROCK_ENABLED=true
BEDROCK_REGION=us-west-2

# Use inference profile IDs (required for Claude Sonnet 4.5)
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_LITE_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_PRO_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**Important**: Claude Sonnet 4.5 requires using **inference profile IDs** (prefix `us.` or `eu.`) instead of direct model IDs. If you see the error:

```
ValidationException: Invocation of model ID with on-demand throughput isn't supported
```

This means you need to update your model IDs to use the inference profile format (`us.anthropic.claude-sonnet-4-5-20250929-v1:0` instead of `anthropic.claude-sonnet-4-5-20250929-v1:0`).

## Troubleshooting

### "Unable to locate credentials"
- AWS credentials not configured
- Follow Option 1 or 2 above

### "Model access denied"
- Request model access in AWS Bedrock console
- Wait for approval

### "Region not supported"
- Bedrock might not be available in your AWS region
- Change `BEDROCK_REGION` to a supported region (e.g., `us-west-2`, `us-east-1`)

### Still showing fallback messages
- Restart the backend server after configuring credentials
- Check server logs for specific error messages
- Verify `BEDROCK_ENABLED=true` in `.env.local`

## Cost Considerations

Claude Sonnet 4.5 on AWS Bedrock charges per token:
- Input: ~$0.003 per 1K tokens
- Output: ~$0.015 per 1K tokens

The application limits token usage:
- `NLQ_MAX_TOKENS=400` for SQL queries
- `max_tokens=300` for general chat responses

Monitor your AWS billing dashboard to track costs.

## Support

For issues:
1. Check `/bedrock_test` endpoint
2. Review server logs for `[PATIENT_CHAT]` messages
3. Verify AWS credentials: `aws sts get-caller-identity`
4. Ensure Bedrock model access is granted

