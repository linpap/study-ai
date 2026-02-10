const INSTAMOJO_CLIENT_ID = process.env.INSTAMOJO_CLIENT_ID;
const INSTAMOJO_CLIENT_SECRET = process.env.INSTAMOJO_CLIENT_SECRET;

const INSTAMOJO_BASE_URL = process.env.INSTAMOJO_SANDBOX === 'true'
  ? 'https://test.instamojo.com'
  : 'https://api.instamojo.com';

export async function getInstamojoToken(): Promise<string> {
  if (!INSTAMOJO_CLIENT_ID || !INSTAMOJO_CLIENT_SECRET) {
    throw new Error('Instamojo credentials not configured');
  }

  const response = await fetch(`${INSTAMOJO_BASE_URL}/oauth2/token/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      client_id: INSTAMOJO_CLIENT_ID,
      client_secret: INSTAMOJO_CLIENT_SECRET,
    }),
  });

  const data = await response.json();

  if (!data.access_token) {
    throw new Error(`Failed to get Instamojo access token: ${JSON.stringify(data)}`);
  }

  return data.access_token;
}

export { INSTAMOJO_BASE_URL };
