from google import genai

client = genai.Client(api_key="AIzaSyCIpIsstpOIQm2fae69c75DWG8VPNaB2OI")

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Hello"
)

print(response.text)