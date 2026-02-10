const INSTAMOJO_API_KEY = process.env.INSTAMOJO_API_KEY;
const INSTAMOJO_AUTH_TOKEN = process.env.INSTAMOJO_AUTH_TOKEN;

const INSTAMOJO_BASE_URL = process.env.INSTAMOJO_SANDBOX === 'true'
  ? 'https://test.instamojo.com/api/1.1'
  : 'https://www.instamojo.com/api/1.1';

export function getInstamojoHeaders(): Record<string, string> {
  if (!INSTAMOJO_API_KEY || !INSTAMOJO_AUTH_TOKEN) {
    throw new Error('Instamojo credentials not configured');
  }

  return {
    'X-Api-Key': INSTAMOJO_API_KEY,
    'X-Auth-Token': INSTAMOJO_AUTH_TOKEN,
  };
}

export { INSTAMOJO_BASE_URL };
