FROM nginx:alpine

WORKDIR /app
COPY index.html .
COPY style.css .

EXPOSE 80  

COPY nginx.conf /etc/nginx/conf.d/default.conf

ARG API_URL
RUN sed -i 's|__API_URL__|'"${API_URL}"'|g' index.html

CMD ["nginx", "-g", "daemon off;"]

# RUN apk add --no-cache sed
# CMD ["sh", "-c", "sed -i 's|__API_URL__|${API_URL}|g' /app/index.html && nginx -g 'daemon off;'"]
