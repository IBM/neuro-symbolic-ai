"use strict";(self.webpackChunkneuro_symbolic_ai_toolkit_site=self.webpackChunkneuro_symbolic_ai_toolkit_site||[]).push([[5068],{4594:function(e,t,r){r.r(t),r.d(t,{_frontmatter:function(){return i},default:function(){return u}});var n=r(3366),a=(r(7294),r(4983)),o=r(9985),l=["components"],i={},c={_frontmatter:i},s=o.Z;function u(e){var t=e.components,r=(0,n.Z)(e,l);return(0,a.kt)(s,Object.assign({},c,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",null,"Description"),(0,a.kt)("p",null,"The logic generated by this component is used by the next stages of the pipeline to learn the policy.  The development repository is ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/CognitiveHorizons/AMR-CSLogic"},"here")," (limited access)."),(0,a.kt)("h2",null,"Main Contributors"),(0,a.kt)("p",null,"Vernon Austel, Jason Liang, Rosario Uceda-Sosa, Masaki Ono, Daiki Kimura"))}u.isMDXComponent=!0},6156:function(e,t,r){r.d(t,{Z:function(){return o}});var n=r(7294),a=r(36),o=function(e){var t=e.date,r=new Date(t);return t?n.createElement(a.X2,{className:"last-modified-date-module--row--XJoYQ"},n.createElement(a.sg,null,n.createElement("div",{className:"last-modified-date-module--text--ogPQF"},"Page last updated: ",r.toLocaleDateString("en-GB",{day:"2-digit",year:"numeric",month:"long"})))):null}},9985:function(e,t,r){r.d(t,{Z:function(){return _}});var n=r(7294),a=r(8650),o=r.n(a),l=r(5444),i=r(4983),c=r(5426),s=r(4311),u=r(808),m=r(8318),d=r(4275),p=r(9851),f=r(2881),g=r(6958),b=r(6156);function v(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return y(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return y(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function y(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var h=function(e){for(var t,r=e.frontmatter,a=r.url,o=[],i=v(r.tags.entries());!(t=i()).done;){var c=t.value,s=c[0],u=c[1];o.push(n.createElement(l.Link,{to:"/repos#"+u,key:s},n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label"},u)," ")))}return n.createElement("div",{className:"bx--grid"},n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1"},"Repository: "),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("a",{href:a,target:"_blank",rel:"noreferrer"},a))),n.createElement("div",{className:"bx--row"},n.createElement("div",{className:"bx--col-lg-1 category-header"},"Categories:"),n.createElement("div",{className:"bx--col-lg-4"},n.createElement("div",{className:"RepoHeader-module--flex--anwdv"},o))))},E=r(6258);function x(e,t){var r="undefined"!=typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(r)return(r=r.call(e)).next.bind(r);if(Array.isArray(e)||(r=function(e,t){if(!e)return;if("string"==typeof e)return k(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return k(e,t)}(e))||t&&e&&"number"==typeof e.length){r&&(e=r);var n=0;return function(){return n>=e.length?{done:!0}:{done:!1,value:e[n++]}}}throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function k(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}var w=function(e,t){var r=function(e,t){var r=t.replace("/repos/","");return e.allMdx.edges.filter((function(e){return e.node.slug===r}))[0].node}(e,t),a=r.frontmatter,o="/repos/"+r.slug,i=n.createElement("div",null,n.createElement("div",{className:E.pb},n.createElement("h4",null,a.title),n.createElement("p",{className:E.pU},a.description)),n.createElement("p",{className:E.pt},function(e){for(var t,r=[],a=x(e);!(t=a()).done;){var o=t.value;r.push(n.createElement("button",{class:"bx--tag bx--tag--green"}," ",n.createElement("span",{class:"bx--tag__label"},o)," "))}return r}(a.tags)));return n.createElement(l.Link,{to:o,className:E.Gg},i)},N=function(e){return n.createElement(l.StaticQuery,{query:"3281138953",render:function(t){return w(t,e.to)}})},S=function(e){return n.createElement("div",{className:E.fU},e.children)},_=function(e){var t=e.pageContext,r=e.children,a=e.location,v=e.Title,y=t.frontmatter,E=void 0===y?{}:y,x=t.relativePagePath,k=t.titleType,w=E.tabs,_=E.title,A=E.theme,Z=E.description,T=E.keywords,C=E.date,L=(0,g.Z)().interiorTheme,M={RepoLink:N,RepoLinkList:S},j=(0,l.useStaticQuery)("2102389209").site.pathPrefix,D=j?a.pathname.replace(j,""):a.pathname,P=w?D.split("/").filter(Boolean).slice(-1)[0]||o()(w[0],{lower:!0}):"",I=A||L;return n.createElement(s.Z,{tabs:w,homepage:!1,theme:I,pageTitle:_,pageDescription:Z,pageKeywords:T,titleType:k},n.createElement(u.Z,{title:v?n.createElement(v,null):_,label:"label",tabs:w,theme:I}),w&&n.createElement(p.Z,{title:_,slug:D,tabs:w,currentTab:P}),n.createElement(f.Z,{padded:!0},n.createElement(h,{frontmatter:E}),n.createElement(i.Zo,{components:M},r),n.createElement(m.Z,{relativePagePath:x}),n.createElement(b.Z,{date:C})),n.createElement(d.Z,{pageContext:t,location:a,slug:D,tabs:w,currentTab:P}),n.createElement(c.Z,null))}}}]);
//# sourceMappingURL=component---src-pages-repos-amr-cslogic-mdx-b734ec2fb031f9308029.js.map