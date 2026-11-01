import handleRequest from "../app/server.mjs";

export const config = {
  maxDuration: 60,
};

export default async function handler(req, res) {
  return handleRequest(req, res);
}
