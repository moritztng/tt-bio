import { EmailMessage } from "cloudflare:email";
import { handle } from "./contact.js";

export default { fetch: (request, env) => handle(request, env, EmailMessage) };
